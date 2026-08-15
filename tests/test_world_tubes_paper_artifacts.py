from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from research_experiments.paper_runner_suite import (
    generate_world_tubes_paper_artifacts as artifact_generator,
)
from research_experiments.paper_runner_suite.generate_world_tubes_paper_artifacts import (
    COST_KEYS,
    LANE_ORDER,
    PUBLICATION_TIMING_KEYS,
    QUALITY_KEYS,
    THEOREM_SOURCE_PATHS,
    TIMING_KEYS,
    build_bundle,
    collect_theorem_evidence,
    resolve_matrix_run_root,
    verify_bundle_dir,
    verify_manuscript_package,
    write_bundle,
)
from research_experiments.paper_runner_suite.world_tubes_theorem_table import (
    verify_table_report,
)
from research_experiments.star_uvt_feature_tubes.projective_variable_camera_closure_death_curve import (
    COMPILED_QUALITY_AVAILABLE,
    COMPILED_QUALITY_UNAVAILABLE,
    COMPILED_ROW_STATUS,
    REQUIRED_IMPLEMENTATION_SOURCE_PATHS,
    UNAVAILABLE_COMPILED_QUALITY_FIELDS,
    UNRESOLVED_ROW_STATUS,
    WORLD_VJP_PARAMETER_NAMES,
    _sha256_json,
    assemble_report,
    default_camera_program,
    default_compiler_contract,
    default_thresholds,
    default_world_fixture,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SCRIPT = (
    ROOT
    / "research_experiments"
    / "paper_runner_suite"
    / "generate_world_tubes_paper_artifacts.py"
)
SOURCE = {
    "repository_commit": "a" * 40,
    "repository_dirty": False,
    "star_uvt_commit": "b" * 40,
    "star_uvt_dirty": False,
}


def test_artifact_generator_cli_bootstraps_repo_imports_without_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--allow-incomplete" in completed.stdout


@pytest.fixture
def stub_retained_artifact_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let formatting tests use compact fixtures; production has no bypass."""

    monkeypatch.setattr(
        artifact_generator,
        "_validate_public_run_deep",
        lambda _expected, _summary, _path: [],
    )
    monkeypatch.setattr(
        artifact_generator,
        "_validate_frozen_artifact_bindings",
        lambda _path, _summary: [],
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _identity(character: str) -> dict[str, object]:
    return {"schema_version": 1, "sha256": character * 64}


def _hashed_contract(schema_version: int, **values: object) -> dict[str, object]:
    payload = {"schema_version": schema_version, **values}
    return {
        **payload,
        "sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def _protocol(path: Path, name: str = "fixture_protocol") -> None:
    _write_json(
        path,
        {
            "name": name,
            "dataset": {
                "sample_id": "fixture_scene",
                "train_cameras": ["cam00", "cam01"],
                "heldout_cameras": ["cam02"],
            },
        },
    )


def _matrix(path: Path, protocol_path: Path, seeds: list[int]) -> None:
    _write_json(
        path,
        {
            "name": "fixture_matrix",
            "runs": [
                {
                    "role": "primary_progressive",
                    "protocol": str(protocol_path),
                    "seeds": seeds,
                    "world_tubes_backward_policy": "fast_exploration",
                }
            ],
        },
    )


def _evidence(offset: float) -> dict[str, object]:
    quality = {
        key: value
        for key, value in zip(
            QUALITY_KEYS,
            (
                20.0 + offset,
                0.8,
                0.1,
                18.0 + offset,
                0.7,
                0.15,
                0.25,
            ),
            strict=True,
        )
    }
    cost_values = (
        10,
        40,
        40,
        1_000,
        1_000,
        100,
        100,
        400,
        800,
        1_024,
        2_048,
        4_096,
        1.0 + offset * 0.01,
    )
    timing_values = (0.1, 0.2, 9, 0.3, 10, 0.05, 10, 1.0 + offset * 0.01)
    return {
        "schema_version": 2,
        "quality": quality,
        "cost": dict(zip(COST_KEYS, cost_values, strict=True)),
        "timing": dict(zip(TIMING_KEYS, timing_values, strict=True)),
        "diagnostics": {"active_count": 100},
    }


def _run_summary(seed: int) -> dict[str, object]:
    source = {
        "repository_commit": "a" * 40,
        "repository_dirty": False,
        "star_uvt_commit": "b" * 40,
        "star_uvt_dirty": False,
    }
    dataset_family = "neural_3d_video"
    pose_source = "neural_3d_llff_opencv_relative_pinhole_v2"
    common = {
        "schema_version": 1,
        "dataset_input_identity": _hashed_contract(
            1,
            dataset=dataset_family,
            files=[],
        ),
        "decoded_dataset_bundle": _hashed_contract(
            1,
            pose_source=pose_source,
        ),
        "evaluator": _identity("3"),
        "runtime": _identity("4"),
        "sample_schedule": _identity(chr(ord("5") + (seed % 2))),
    }
    lanes = {}
    for lane_index, lane_name in enumerate(LANE_ORDER):
        lanes[lane_name] = {
            "evidence": _evidence(float(1 - lane_index)),
            "route_native_extension": {"sha256": chr(ord("c") + lane_index) * 64},
            "wandb": {
                "mode": "offline",
                "run_id": f"{lane_name}-{seed}",
                "run_file": {"sha256": chr(ord("6") + lane_index) * 64},
            },
            "paper_protocol": {
                "paper_dataset_bundle": common["decoded_dataset_bundle"],
                "paper_evaluator": common["evaluator"],
                "paper_runtime": common["runtime"],
                "sample_schedule": common["sample_schedule"],
            },
        }
    return {
        "status": "complete",
        "seed": seed,
        "protocol": {
            "name": "fixture_protocol",
            "dataset": {
                "sample_id": "fixture_scene",
                "train_cameras": ["cam00", "cam01"],
                "heldout_cameras": ["cam02"],
            },
        },
        "world_tubes_requested_backward_policy": "fast_exploration",
        "world_tubes_backward_policy": "fast_exploration",
        "source": source,
        "source_finish": source,
        "manifest_validation": {
            "dataset": dataset_family,
            "expected_pose_source": pose_source,
            "input_identity": common["dataset_input_identity"],
        },
        "common_evidence_contract": common,
        "lanes": lanes,
    }


def _theorem_report() -> dict[str, object]:
    return json.loads(
        (
            ROOT
            / "outputs"
            / "benchmarks"
            / "2026-07-22_world_tubes_theorem_table"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )


def _timing_fixture_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return (
            ordered[lower] * (1.0 - fraction)
            + ordered[upper] * fraction
        )

    return {
        "count": len(ordered),
        "min": min(ordered),
        "p25": quantile(0.25),
        "median": statistics.median(ordered),
        "p75": quantile(0.75),
        "max": max(ordered),
        "mean": statistics.fmean(ordered),
    }


def _frozen_row(frame_count: int) -> dict[str, object]:
    multipliers = (0.96, 0.98, 1.0, 1.02, 1.04)
    replay_forward = [0.001 * frame_count * value for value in multipliers]
    replay_backward = [0.002 * frame_count * value for value in multipliers]
    compiled_compile = [0.003 * value for value in multipliers]
    compiled_forward = [0.0004 * frame_count * value for value in multipliers]
    compiled_backward = [0.0006 * frame_count * value for value in multipliers]
    timing_samples = {
        "replay_total_forward": replay_forward,
        "replay_total_backward": replay_backward,
        "replay_total_forward_backward": [
            forward + backward
            for forward, backward in zip(
                replay_forward,
                replay_backward,
                strict=True,
            )
        ],
        "replay_per_frame_forward": [
            value / frame_count for value in replay_forward
        ],
        "replay_per_frame_backward": [
            value / frame_count for value in replay_backward
        ],
        "compiled_atlas_compile": compiled_compile,
        "compiled_total_forward": compiled_forward,
        "compiled_total_backward": compiled_backward,
        "compiled_total_forward_backward": [
            forward + backward
            for forward, backward in zip(
                compiled_forward,
                compiled_backward,
                strict=True,
            )
        ],
        "compiled_compile_plus_forward_backward": [
            compile_time + forward + backward
            for compile_time, forward, backward in zip(
                compiled_compile,
                compiled_forward,
                compiled_backward,
                strict=True,
            )
        ],
        "compiled_per_frame_forward": [
            value / frame_count for value in compiled_forward
        ],
        "compiled_per_frame_backward": [
            value / frame_count for value in compiled_backward
        ],
    }
    timing_summary = {
        key: _timing_fixture_summary(values)
        for key, values in timing_samples.items()
    }
    assert set(timing_samples) == set(PUBLICATION_TIMING_KEYS)
    return {
        "schema_version": 2,
        "status": "complete",
        "accepted": True,
        "frame_count": frame_count,
        "image": {"max_abs_error": 1.0e-7, "mean_abs_error": 1.0e-8},
        "gradient": {
            "global_normalized_l2_error": 1.0e-7,
            "max_parameter_normalized_l2_error": 2.0e-7,
        },
        "atlas": {
            "trace_count": 100,
            "cell_count": 20,
            "interval_trace_entries": 200,
            "fallback_fraction": 0.0,
        },
        "payload_bytes": {
            "metric_kind": "logical_work_volume_proxy",
            "topology_bytes_included": False,
            "storage_claim_eligible": False,
            "publication_claim_eligible": False,
            "compiled_to_replay_logical_volume_ratio": 4.0 / frame_count,
        },
        "retained_storage_bytes": {
            "topology_bytes_included": True,
            "storage_claim_eligible": True,
            "publication_claim_eligible": True,
        },
        "route_memory": {
            "compiled_parity_replay_excluded": True,
            "publication_claim_eligible": True,
        },
        "timing_benchmark": {
            "schema_version": 1,
            "status": "complete",
            "label": "warmed_repeated_wall_timing_v1",
            "publication_ready": True,
            "warmups": 1,
            "repeats": 5,
            "samples_s": timing_samples,
            "summary_s": timing_summary,
        },
    }


def _frozen_report() -> dict[str, object]:
    source = {
        "repository_commit": "a" * 40,
        "repository_dirty": False,
        "star_uvt_commit": "b" * 40,
        "star_uvt_dirty": False,
    }
    frame_counts = [4, 8, 16, 32, 64, 128, 300]
    return {
        "schema_version": 1,
        "status": "accepted",
        "publication_eligible": True,
        "timing_warmups": 1,
        "timing_repeats": 5,
        "source": source,
        "source_finish": source,
        "protocol": {"name": "fixture_protocol"},
        "seed": 17,
        "frozen_world_replay_compiled_sweep": {
            "schema_version": 1,
            "status": "complete",
            "requested_frame_counts": [0, 4, 8, 16, 32, 64, 128],
            "full_dataset_frame_count": 300,
            "all_rows_accepted": True,
            "publication_eligible": True,
            "selected_time_slice_parity_accepted": True,
            "all_rows_timing_publication_ready": True,
            "all_rows_storage_publication_ready": True,
            "all_rows_route_memory_publication_ready": True,
            "timing_benchmark_warmups": 1,
            "timing_benchmark_repeats": 5,
            "checkpoint_shared_across_rows": True,
            "world_state_shared_across_rows": True,
            "rows": [_frozen_row(frame_count) for frame_count in frame_counts],
        },
    }


HALF_SPANS = [15.0, 45.0, 75.0, 120.0]
VARIABLE_FRAMES = 64
VARIABLE_IMAGE_SIZE = 64


def _variable_row(
    *,
    span: float,
    chart_count: int,
    dead: bool,
) -> dict[str, object]:
    primitive_count = 3
    projected_samples = primitive_count * VARIABLE_FRAMES
    trace_count = primitive_count * chart_count
    cell_count = 4 * chart_count
    interval_entries = 2 * trace_count
    dense_samples = 8 * interval_entries
    image_mse = 1.0e-3 if dead else 1.0e-8
    vjp_error = 0.05 if dead else 0.005
    q_extent = math.tan(0.5 * math.radians(span))
    return {
        "row_status": COMPILED_ROW_STATUS,
        "row_scope": "compiled_quality_closure_or_threshold_death",
        "compiled_quality_metrics_status": COMPILED_QUALITY_AVAILABLE,
        "compiled_quality_metrics_unavailable": [],
        "motion_half_span_degrees": span,
        "motion_total_span_degrees": 2.0 * span,
        "physical_interval": [-1.0, 1.0],
        "sample_count": VARIABLE_FRAMES,
        "q_min": -q_extent,
        "q_max": q_extent,
        "chart_count": chart_count,
        "accepted_chart_count": chart_count,
        "unresolved_chart_count": 0,
        "unresolved_chart_reasons": [],
        "unresolved_charts": [],
        "unresolved_chart_metadata_sha256": _sha256_json([]),
        "accepted_chart_fraction": 1.0,
        "sampled_max_fit_residual_uv_px": 0.1,
        "fit_residual_semantics": "empirical_max_over_requested_samples",
        "min_denominator_abs": 1.0,
        "support_event_count": chart_count,
        "visibility_event_count": 0,
        "event_count": chart_count,
        "event_interval_count": chart_count + 1,
        "reference_support_policy": "full_image",
        "reference_order_policy": "all_live_depth_per_sample",
        "reference_fallback_reason": "oracle_all_live_depth_sort",
        "reference_sample_semantics": "empirical_at_requested_samples",
        "reference_cell_count": cell_count,
        "reference_live_sorted_cell_count": cell_count,
        "trace_count": trace_count,
        "trace_to_replay_ratio": trace_count / projected_samples,
        "cell_count": cell_count,
        "visibility_stratum_split_cell_count": 0,
        "interval_entry_count": interval_entries,
        "dense_trace_samples": dense_samples,
        "interval_to_dense_ratio": interval_entries / dense_samples,
        "fallback_cell_count": 0,
        "fallback_cell_fraction": 0.0,
        "fallback_trace_samples": 0,
        "fallback_sample_fraction": 0.0,
        "fallback_reasons": [],
        "invalid_sample_count": 0,
        "projected_sample_count": projected_samples,
        "invalid_sample_fraction": 0.0,
        "post_visibility_stale": False,
        "post_order_mismatch_sample_count": 0,
        "post_ambiguous_depth_sample_count": 0,
        "image_mse": image_mse,
        "image_psnr_db": 10.0 * math.log10(1.0 / image_mse),
        "image_p999_abs_error": 0.05 if dead else 0.001,
        "image_max_abs_error": 0.08 if dead else 0.002,
        "world_vjp_rel_l2_by_parameter": {
            name: vjp_error
            for name in WORLD_VJP_PARAMETER_NAMES
        },
        "world_vjp_rel_l2_max": vjp_error,
        "world_vjp_parameter_names": list(WORLD_VJP_PARAMETER_NAMES),
        "world_vjp_reference_norm_by_parameter": {
            name: 1.0
            for name in WORLD_VJP_PARAMETER_NAMES
        },
        "world_vjp_compiled_norm_by_parameter": {
            name: 1.0
            for name in WORLD_VJP_PARAMETER_NAMES
        },
        "world_vjp_nonzero_parameter_count": 5,
        "vjp_topology_semantics": (
            "fixed_compiled_topology_away_from_event_boundaries"
        ),
    }


def _variable_unresolved_row(*, span: float, chart_count: int) -> dict[str, object]:
    q_extent = math.tan(0.5 * math.radians(span))
    unresolved = [
        {
            "start": VARIABLE_FRAMES - 2,
            "stop": VARIABLE_FRAMES,
            "sample_count": 2,
            "reason": "depth_residual",
            "reasons": ["depth_residual"],
            "sampled_max_fit_residual_uv_px": 0.1,
            "sampled_max_fit_residual_depth": 0.04,
            "min_denominator_abs": 1.0,
            "denominator_root_count": 0,
            "min_valid_fraction": 1.0,
        }
    ]
    return {
        "row_status": UNRESOLVED_ROW_STATUS,
        "row_scope": "death_boundary_only_not_closure_evidence",
        "compiled_quality_metrics_status": COMPILED_QUALITY_UNAVAILABLE,
        "compiled_quality_metrics_unavailable": list(UNAVAILABLE_COMPILED_QUALITY_FIELDS),
        "compiler_failure_class": "trace_window_certificate_unsatisfied",
        "motion_half_span_degrees": span,
        "motion_total_span_degrees": 2.0 * span,
        "physical_interval": [-1.0, 1.0],
        "sample_count": VARIABLE_FRAMES,
        "q_min": -q_extent,
        "q_max": q_extent,
        "chart_count": chart_count,
        "accepted_chart_count": chart_count - 1,
        "unresolved_chart_count": 1,
        "accepted_chart_fraction": (chart_count - 1) / chart_count,
        "fit_residual_semantics": "empirical_max_over_requested_samples",
        "unresolved_chart_reasons": ["depth_residual"],
        "unresolved_charts": unresolved,
        "unresolved_chart_metadata_sha256": _sha256_json(unresolved),
        "unresolved_max_fit_residual_uv_px": 0.1,
        "unresolved_max_fit_residual_depth": 0.04,
        "unresolved_min_denominator_abs": 1.0,
        "unresolved_min_valid_fraction": 1.0,
        "unresolved_denominator_root_count": 0,
        "projected_sample_count": 3 * VARIABLE_FRAMES,
    }


def _variable_report() -> dict[str, object]:
    rows = [
        _variable_row(span=HALF_SPANS[0], chart_count=2, dead=False),
        _variable_row(span=HALF_SPANS[1], chart_count=3, dead=False),
        _variable_row(span=HALF_SPANS[2], chart_count=5, dead=True),
        _variable_unresolved_row(span=HALF_SPANS[3], chart_count=8),
    ]
    source_files = [
        {
            "path": path,
            "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
        }
        for path in REQUIRED_IMPLEMENTATION_SOURCE_PATHS
    ]
    return assemble_report(
        rows,
        half_spans_degrees=HALF_SPANS,
        world_fixture=default_world_fixture(),
        camera_program=default_camera_program(
            frames=VARIABLE_FRAMES,
            image_size=VARIABLE_IMAGE_SIZE,
        ),
        compiler=default_compiler_contract(
            tile_size=16,
            sigma_px=1.6,
            support_padding_px=6.0,
            max_residual_uv=0.25,
            max_depth_residual=0.025,
            min_denominator_abs=1.0e-3,
            max_windows=256,
        ),
        thresholds=default_thresholds(),
        runtime={"device": "cpu", "torch_version": "fixture"},
        implementation={
            "source_files": source_files,
            "source_manifest_sha256": _sha256_json(source_files),
        },
        source=dict(SOURCE),
        source_finish=dict(SOURCE),
        dirty_source_allowed=False,
    )


def _complete_inputs(tmp_path: Path) -> dict[str, Any]:
    protocol = tmp_path / "protocol.json"
    matrix = tmp_path / "matrix.json"
    run_root = tmp_path / "runs"
    matrix_summary = run_root / "matrix_summary.json"
    theorem = tmp_path / "theorem.json"
    frozen = tmp_path / "frozen.json"
    variable = tmp_path / "variable.json"
    _protocol(protocol)
    _matrix(matrix, protocol, [17, 29])
    matrix_records = []
    for seed in (17, 29):
        summary = _run_summary(seed)
        summary_path = (
            run_root
            / "fixture_protocol"
            / f"seed_{seed}"
            / "run_summary.json"
        )
        _write_json(
            summary_path,
            summary,
        )
        matrix_records.append(
            {
                "run": {
                    "key": (
                        f"fixture_protocol/seed_{seed}/fast_exploration"
                    ),
                    "role": "primary_progressive",
                    "protocol": str(protocol.resolve()),
                    "seed": seed,
                    "world_tubes_backward_policy": "fast_exploration",
                    "worldfoam_initializer": "base_config",
                },
                "summary": {
                    **summary,
                    "run_summary_path": str(summary_path.resolve()),
                },
            }
        )
    _write_json(
        matrix_summary,
        {
            "status": "complete",
            "matrix": "fixture_matrix",
            "run_count": 2,
            "lane_row_count": 6,
            "runs": matrix_records,
            "artifacts": {},
        },
    )
    _write_json(theorem, _theorem_report())
    _write_json(frozen, _frozen_report())
    _write_json(variable, _variable_report())
    return {
        "matrix_path": matrix,
        "run_root": run_root,
        "matrix_summary": matrix_summary,
        "theorem_summary": theorem,
        "frozen_summary": frozen,
        "variable_camera_summary": variable,
        "verify_current_variable_camera_source": False,
    }


def test_incomplete_bundle_withholds_partial_numeric_artifacts(
    tmp_path: Path,
    stub_retained_artifact_validation: None,
) -> None:
    inputs = _complete_inputs(tmp_path)
    missing = (
        inputs["run_root"]
        / "fixture_protocol"
        / "seed_29"
        / "run_summary.json"
    )
    missing.unlink()

    bundle = build_bundle(**inputs)
    out_dir = tmp_path / "bundle"
    write_bundle(bundle, out_dir)

    assert bundle["submission_ready"] is False
    assert bundle["components"]["public_context"]["accepted_run_count"] == 1
    assert bundle["components"]["public_context"]["rows"] == []
    assert "NOT SUBMISSION-READY" in (
        out_dir / "public_context_table.md"
    ).read_text(encoding="utf-8")
    assert 'class="data-bar"' not in (
        out_dir / "public_heldout_quality.svg"
    ).read_text(encoding="utf-8")
    assert verify_bundle_dir(out_dir, require_complete=False) == []
    assert "artifact evidence bundle is not complete" in "\n".join(
        verify_bundle_dir(out_dir, require_complete=True)
    )


def test_complete_bundle_is_deterministic_and_manuscript_consumable(
    tmp_path: Path,
    stub_retained_artifact_validation: None,
) -> None:
    inputs = _complete_inputs(tmp_path)
    bundle = build_bundle(**inputs)
    first = tmp_path / "bundle_a"
    second = tmp_path / "bundle_b"
    write_bundle(bundle, first)
    write_bundle(bundle, second)

    assert bundle["status"] == "complete"
    assert bundle["submission_ready"] is True
    assert bundle["readiness_scope"] == "evidence_artifact_bundle_only"
    assert bundle["manuscript_package_required"] is True
    assert verify_bundle_dir(first) == []
    assert verify_bundle_dir(second) == []
    assert {path.name for path in first.iterdir()} == {
        path.name for path in second.iterdir()
    }
    for first_path in first.iterdir():
        assert first_path.read_bytes() == (second / first_path.name).read_bytes()
    assert "19.000" in (first / "public_context_table.md").read_text(
        encoding="utf-8"
    )
    assert "begin{tabular}" in (first / "frozen_scaling_table.tex").read_text(
        encoding="utf-8"
    )
    assert r"\leq" in (first / "theorem_table.tex").read_text(encoding="utf-8")
    assert 'class="data-line"' in (first / "frozen_scaling.svg").read_text(
        encoding="utf-8"
    )


def test_schema_v1_public_row_is_rejected_without_exporting_numbers(
    tmp_path: Path,
    stub_retained_artifact_validation: None,
) -> None:
    inputs = _complete_inputs(tmp_path)
    summary_path = (
        inputs["run_root"]
        / "fixture_protocol"
        / "seed_17"
        / "run_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["lanes"]["world_tubes"]["evidence"]["schema_version"] = 1
    _write_json(summary_path, summary)
    matrix_summary = json.loads(
        inputs["matrix_summary"].read_text(encoding="utf-8")
    )
    embedded = matrix_summary["runs"][0]["summary"]
    retained_path = embedded["run_summary_path"]
    matrix_summary["runs"][0]["summary"] = {
        **summary,
        "run_summary_path": retained_path,
    }
    _write_json(inputs["matrix_summary"], matrix_summary)

    bundle = build_bundle(**inputs)
    public = bundle["components"]["public_context"]

    assert public["status"] == "invalid"
    assert public["rows"] == []
    rejected = next(slot for slot in public["slots"] if slot["seed"] == 17)
    assert rejected["status"] == "rejected"
    assert any("schema 2" in error for error in rejected["errors"])


def test_public_artifacts_reject_hash_consistent_legacy_neural3d_pose_source(
    tmp_path: Path,
    stub_retained_artifact_validation: None,
) -> None:
    inputs = _complete_inputs(tmp_path)
    legacy_pose_source = "neural_3d_llff_relative_pinhole"

    def replace_pose_source(summary: dict[str, object]) -> None:
        common = summary["common_evidence_contract"]
        bundle = _hashed_contract(1, pose_source=legacy_pose_source)
        common["decoded_dataset_bundle"] = bundle
        for lane in summary["lanes"].values():
            lane["paper_protocol"]["paper_dataset_bundle"] = bundle

    summary_path = (
        inputs["run_root"]
        / "fixture_protocol"
        / "seed_17"
        / "run_summary.json"
    )
    retained = json.loads(summary_path.read_text(encoding="utf-8"))
    replace_pose_source(retained)
    _write_json(summary_path, retained)
    matrix_summary = json.loads(
        inputs["matrix_summary"].read_text(encoding="utf-8")
    )
    replace_pose_source(matrix_summary["runs"][0]["summary"])
    _write_json(inputs["matrix_summary"], matrix_summary)

    public = build_bundle(**inputs)["components"]["public_context"]

    assert public["status"] == "invalid"
    assert public["rows"] == []
    rejected = next(slot for slot in public["slots"] if slot["seed"] == 17)
    assert any(
        "decoded dataset pose source does not match" in error
        for error in rejected["errors"]
    )


@pytest.mark.parametrize(
    "mutation",
    ("run_key", "embedded_summary"),
)
def test_public_evidence_requires_exact_canonical_matrix_binding(
    tmp_path: Path,
    mutation: str,
    stub_retained_artifact_validation: None,
) -> None:
    inputs = _complete_inputs(tmp_path)
    matrix_summary = json.loads(
        inputs["matrix_summary"].read_text(encoding="utf-8")
    )
    if mutation == "run_key":
        matrix_summary["runs"][0]["run"]["key"] = "drifted"
    else:
        matrix_summary["runs"][0]["summary"]["lanes"]["world_tubes"][
            "evidence"
        ]["quality"]["heldout_eval_psnr"] += 1.0
    _write_json(inputs["matrix_summary"], matrix_summary)

    bundle = build_bundle(**inputs)
    public = bundle["components"]["public_context"]

    assert public["status"] == "invalid"
    assert public["rows"] == []
    all_errors = [
        error
        for slot in public["slots"]
        for error in slot["errors"]
    ]
    expected_fragment = (
        "run keys/order"
        if mutation == "run_key"
        else "does not exactly match retained"
    )
    assert any(expected_fragment in error for error in all_errors)


def test_public_evidence_invokes_matrix_runner_deep_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _complete_inputs(tmp_path)
    calls: list[Path] = []

    monkeypatch.setattr(
        artifact_generator.matrix_runner,
        "resolve_paper_training_protocol",
        lambda _raw: object(),
    )

    def reject_tampered_retained_artifact(
        _run: object,
        _summary: object,
        *,
        protocol: object,
        summary_path: Path,
    ) -> None:
        assert protocol is not None
        calls.append(summary_path)
        raise ValueError("tampered retained child artifact")

    monkeypatch.setattr(
        artifact_generator.matrix_runner,
        "validate_existing_summary",
        reject_tampered_retained_artifact,
    )
    public = artifact_generator.collect_public_evidence(
        inputs["matrix_path"],
        inputs["run_root"],
        inputs["matrix_summary"],
    )

    assert len(calls) == 2
    assert public["status"] == "invalid"
    assert public["rows"] == []
    assert all(
        any("tampered retained child artifact" in error for error in slot["errors"])
        for slot in public["slots"]
    )


@pytest.mark.parametrize("tampered_name", ("comparison_report", "execution_identity"))
def test_frozen_evidence_rejects_tampered_bound_sidecar(
    tmp_path: Path,
    tampered_name: str,
) -> None:
    summary_path = tmp_path / "summary.json"
    comparison_path = tmp_path / "comparison_report.json"
    identity_path = tmp_path / "execution_identity.json"
    protocol_path = artifact_generator.frozen_runner.DEFAULT_PROTOCOL
    _write_json(comparison_path, {"artifact": "comparison"})
    _write_json(identity_path, {"artifact": "identity"})
    report = _frozen_report()
    report.update(
        {
            "protocol_path": str(protocol_path),
            "protocol_sha256": hashlib.sha256(
                protocol_path.read_bytes()
            ).hexdigest(),
            "comparison_report": str(comparison_path),
            "comparison_report_sha256": hashlib.sha256(
                comparison_path.read_bytes()
            ).hexdigest(),
            "execution_identity": str(identity_path),
            "execution_identity_sha256": hashlib.sha256(
                identity_path.read_bytes()
            ).hexdigest(),
        }
    )
    tampered_path = (
        comparison_path
        if tampered_name == "comparison_report"
        else identity_path
    )
    _write_json(tampered_path, {"artifact": "tampered"})
    _write_json(summary_path, report)

    component = artifact_generator.collect_frozen_evidence(summary_path)

    assert component["status"] == "invalid"
    assert component["rows"] == []
    assert any(
        f"{tampered_name}.json hash binding drifted" in error
        for error in component["errors"]
    )


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    (
        ("short_samples", "exactly 5 finite nonnegative values"),
        ("stale_summary", "does not match raw samples"),
        ("derived_identity", "violates its derived identity"),
    ),
)
def test_frozen_timing_recomputes_raw_sample_contract(
    tmp_path: Path,
    mutation: str,
    expected_fragment: str,
    stub_retained_artifact_validation: None,
) -> None:
    inputs = _complete_inputs(tmp_path)
    frozen = json.loads(inputs["frozen_summary"].read_text(encoding="utf-8"))
    timing = frozen["frozen_world_replay_compiled_sweep"]["rows"][0][
        "timing_benchmark"
    ]
    if mutation == "short_samples":
        timing["samples_s"]["replay_total_forward"].pop()
    elif mutation == "stale_summary":
        timing["summary_s"]["compiled_atlas_compile"]["median"] += 1.0
    else:
        key = "compiled_total_forward_backward"
        timing["samples_s"][key][0] += 1.0
        timing["summary_s"][key] = _timing_fixture_summary(
            timing["samples_s"][key]
        )
    _write_json(inputs["frozen_summary"], frozen)

    bundle = build_bundle(**inputs)
    component = bundle["components"]["frozen_world_scaling"]

    assert component["status"] == "invalid"
    assert component["rows"] == []
    assert any(expected_fragment in error for error in component["errors"])


def test_bundle_verifier_detects_post_generation_drift(
    tmp_path: Path,
    stub_retained_artifact_validation: None,
) -> None:
    inputs = _complete_inputs(tmp_path)
    out_dir = tmp_path / "bundle"
    write_bundle(build_bundle(**inputs), out_dir)
    table_path = out_dir / "public_context_table.md"
    table_path.write_text(
        table_path.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )

    errors = verify_bundle_dir(out_dir)

    assert any("public_context_table.md" in error for error in errors)


def test_variable_camera_evidence_requires_current_implementation(
    tmp_path: Path,
    stub_retained_artifact_validation: None,
) -> None:
    inputs = _complete_inputs(tmp_path)
    report = json.loads(
        inputs["variable_camera_summary"].read_text(encoding="utf-8")
    )
    report["implementation"]["source_files"][0]["sha256"] = "0" * 64
    report["implementation"]["source_manifest_sha256"] = _sha256_json(
        report["implementation"]["source_files"]
    )
    _write_json(inputs["variable_camera_summary"], report)
    inputs["verify_current_variable_camera_source"] = True

    component = build_bundle(**inputs)["components"][
        "variable_camera_closure_death"
    ]

    assert component["status"] == "invalid"
    assert component["rows"] == []
    assert any(
        "implementation source hash mismatch" in error
        for error in component["errors"]
    )


@pytest.mark.parametrize("mutation", ("value", "timing_row"))
def test_theorem_evidence_is_rederived_and_excludes_fixture_timing(
    tmp_path: Path,
    mutation: str,
    stub_retained_artifact_validation: None,
) -> None:
    inputs = _complete_inputs(tmp_path)
    report = json.loads(inputs["theorem_summary"].read_text(encoding="utf-8"))
    if mutation == "value":
        report["rows"][0]["value"] *= 2.0
    else:
        report["rows"].append(
            {
                "acceptance": "< 0.5",
                "claim": "fixture timing smuggle",
                "metric": "fixed/replay forward ratio",
                "source": "scaling",
                "value": 0.1,
            }
        )
        report["summary"]["row_count"] += 1
    _write_json(inputs["theorem_summary"], report)

    component = build_bundle(**inputs)["components"]["theorem_correctness"]

    assert component["status"] == "invalid"
    assert component["rows"] == []
    assert any(
        "do not exactly match pinned retained source reports" in error
        for error in component["errors"]
    )


def test_theorem_evidence_binds_exact_retained_source_bytes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_root"
    for relative_path in THEOREM_SOURCE_PATHS.values():
        source = ROOT / relative_path
        destination = source_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    tampered_path = source_root / THEOREM_SOURCE_PATHS["gauge_value"]
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["summary"]["max_rel_error"] *= 2.0
    _write_json(tampered_path, tampered)
    theorem_summary = tmp_path / "theorem_summary.json"
    _write_json(theorem_summary, _theorem_report())

    component = collect_theorem_evidence(
        theorem_summary,
        source_root=source_root,
    )

    assert component["status"] == "invalid"
    assert component["rows"] == []
    assert any(
        "theorem source hash mismatch" in error
        for error in component["errors"]
    )


def test_theorem_report_verifier_rebuilds_current_source_evidence() -> None:
    tampered = _theorem_report()
    tampered["rows"][0]["value"] *= 2.0

    with pytest.raises(ValueError, match="current verified source reports"):
        verify_table_report(tampered)


def test_artifact_run_root_follows_selected_matrix_output_root(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "matrix.json"
    _write_json(
        matrix,
        {
            "name": "alternate_matrix",
            "output_root": "outputs/benchmarks/alternate_schema2",
            "runs": [],
        },
    )

    assert resolve_matrix_run_root(matrix, None) == (
        ROOT / "outputs" / "benchmarks" / "alternate_schema2"
    ).resolve()
    assert resolve_matrix_run_root(
        matrix,
        Path("outputs/benchmarks/explicit"),
    ) == (ROOT / "outputs" / "benchmarks" / "explicit").resolve()


def test_current_manuscript_package_is_honest_while_evidence_is_incomplete(
    tmp_path: Path,
) -> None:
    paper_dir = (
        ROOT
        / "research_notes"
        / "gauged_uvt_trace_atlas"
        / "paper"
    )
    bundle_dir = paper_dir / "generated" / "schema_v2"

    assert verify_manuscript_package(
        bundle_dir=bundle_dir,
        require_complete=False,
    ) == []

    draft = tmp_path / "WORLD_TUBES_PAPER_DRAFT.md"
    shutil.copyfile(paper_dir / draft.name, draft)
    draft.write_text(
        draft.read_text(encoding="utf-8") + "\nlegacy 5.9153\n",
        encoding="utf-8",
    )
    errors = verify_manuscript_package(
        bundle_dir=bundle_dir,
        draft_path=draft,
        require_complete=False,
    )

    assert any("forbidden stale evidence" in error for error in errors)
