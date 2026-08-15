from __future__ import annotations

"""Build the deterministic, fail-closed World Tubes paper artifact bundle.

This module launches no renderer or training workload.  It reuses the paper
runners' retained-artifact validators, which reopen the exact JSON sidecars,
checkpoints, dataset inputs, native binaries/source trees, and current source:

* a completed canonical matrix ``matrix_summary.json`` and its exact retained
  schema-v2 unified paper ``run_summary.json`` files;
* the accepted frozen-world replay/compiled wrapper summary;
* the verified variable-camera closure/death report; and
* the theorem-table summary, rederived from byte-pinned retained reports.

The default command writes an evidence ledger even when inputs are missing,
but exits non-zero and replaces every incomplete numeric table/plot with an
explicit placeholder.  ``--allow-incomplete`` is for checking or packaging
that honest placeholder state; it never promotes partial numbers.

Example:

    PYTHONPATH=src/train .venv/bin/python \
      research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
      --allow-incomplete

Once all inputs exist, omit ``--allow-incomplete``.  Verify a written bundle
without touching any runtime artifact:

    python3 research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py \
      --verify-dir research_notes/gauged_uvt_trace_atlas/paper/generated/schema_v2

Use ``--verify-manuscript`` to check the generated fragments, manuscript
inputs, citations, local media, and stale-number exclusions together.
"""

import argparse
import csv
import hashlib
import html
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SRC = ROOT / "src" / "train"
VARIABLE_CAMERA_REPORTS = (
    ROOT / "research_experiments" / "star_uvt_feature_tubes"
)
for import_root in (ROOT, TRAIN_SRC, VARIABLE_CAMERA_REPORTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from config_utils import load_config_file  # noqa: E402
from paper_training_types import expected_paper_pose_source  # noqa: E402
from projective_variable_camera_closure_death_curve import (  # noqa: E402
    verify_current_implementation,
    verify_variable_camera_closure_death_curve,
)
from research_experiments.paper_runner_suite import (  # noqa: E402
    run_frozen_world_replay_compiled as frozen_runner,
)
from research_experiments.paper_runner_suite import (  # noqa: E402
    run_unified_paper_ablation as single_runner,
)
from research_experiments.paper_runner_suite import (  # noqa: E402
    run_unified_paper_matrix as matrix_runner,
)


GENERATOR_NAME = "world_tubes_paper_artifacts"
GENERATOR_SCHEMA_VERSION = 1
PAPER_EVIDENCE_SCHEMA_VERSION = 2
LANE_ORDER = ("world_tubes", "worldfoam", "dynamic_3dgs")
DEFAULT_WORLDFOAM_INITIALIZER = "base_config"
CANONICAL_FROZEN_FRAME_COUNTS = (4, 8, 16, 32, 64, 128)
PUBLICATION_TIMING_KEYS = (
    "replay_total_forward",
    "replay_total_backward",
    "replay_total_forward_backward",
    "compiled_atlas_compile",
    "compiled_total_forward",
    "compiled_total_backward",
    "compiled_total_forward_backward",
    "compiled_compile_plus_forward_backward",
    "replay_per_frame_forward",
    "replay_per_frame_backward",
    "compiled_per_frame_forward",
    "compiled_per_frame_backward",
)
QUALITY_KEYS = (
    "eval_psnr",
    "eval_ssim",
    "eval_l1",
    "heldout_eval_psnr",
    "heldout_eval_ssim",
    "heldout_eval_l1",
    "heldout_eval_lpips",
)
COST_KEYS = (
    "optimizer_steps",
    "target_frames",
    "rasterized_frames",
    "target_pixels",
    "rasterized_pixels",
    "parameter_count",
    "trainable_parameter_count",
    "parameter_bytes",
    "optimizer_state_bytes",
    "serialized_checkpoint_bytes",
    "sampled_peak_current_allocated_bytes",
    "sampled_peak_driver_allocated_bytes",
    "elapsed_s",
)
TIMING_KEYS = (
    "cold_compile_forward_s",
    "steady_forward_s",
    "steady_forward_calls",
    "backward_s",
    "backward_calls",
    "optimizer_s",
    "optimizer_calls",
    "train_wall_s",
)
AGGREGATE_METRIC_KEYS = (
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

DEFAULT_MATRIX = (
    ROOT
    / "src"
    / "train_configs"
    / "paper_protocols"
    / "world_tubes_submission_matrix_v1.jsonc"
)
DEFAULT_RUN_ROOT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-07-28_world_tubes_submission_matrix_schema2"
)
DEFAULT_FROZEN_SUMMARY = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "world_tubes_frozen_world_replay_compiled_v1"
    / "coffee_martini_full_300f_progressive_512_v1"
    / "seed_17"
    / "summary.json"
)
DEFAULT_VARIABLE_CAMERA_SUMMARY = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-07-28_world_tubes_variable_camera_closure_death_curve"
    / "summary.json"
)
DEFAULT_THEOREM_SUMMARY = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-07-22_world_tubes_theorem_table"
    / "summary.json"
)
THEOREM_SOURCE_PATHS = {
    "gauge_value": (
        "outputs/benchmarks/"
        "2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.json"
    ),
    "gauge_gradient": (
        "outputs/benchmarks/"
        "2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.json"
    ),
    "decisive_demo": (
        "outputs/benchmarks/"
        "2026-07-08_star_uvt_projective_decisive_demo_fixture/summary.json"
    ),
    "visibility": (
        "outputs/benchmarks/"
        "2026-07-08_star_uvt_projective_visibility_stress_suite/summary.json"
    ),
    "exposure": (
        "outputs/benchmarks/"
        "2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.json"
    ),
    "exposure_backward": (
        "outputs/benchmarks/"
        "2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.json"
    ),
    "mixed_fallback_backward": (
        "outputs/benchmarks/"
        "2026-05-25_star_uvt_projective_exposure_rolling_"
        "mixed_fallback_backward/summary.json"
    ),
    "scaling": (
        "outputs/benchmarks/"
        "2026-07-22_world_tubes_same_representation_"
        "scaling_f4_128_cap256/summary.json"
    ),
}
# These are the exact verifier-accepted retained reports used for the bounded
# correctness table.  Pinning them lets this Torch-free submission generator
# reopen and rederive the table without importing the Torch-based report
# modules.  A deliberate evidence refresh must update these digests.
THEOREM_SOURCE_SHA256 = {
    "gauge_value": "8bf44e486a0c787d2e5878ee6d7dc359e88dbf6b56474b7123e29e60150e022a",
    "gauge_gradient": "285d56dfa0e99a11cdd8d2ace7870db589d1006284702da0897dae1dff590f9e",
    "decisive_demo": "94f0450919fd6edefeb07e1ae9708e6e4841575a8a8de894191b6a8543a11ee7",
    "visibility": "64baed1536144f28fa8f3ff79ba4f74ce28e6f5a7917cdef98d6958f064e1115",
    "exposure": "cbe341a363bd941ddecc934951d5afa5f291cfceb0909f39e00f6629f0194324",
    "exposure_backward": "0bd320b7987d34bd767dc3b9396171f637351d45e72bfcb55254ad9e17e8849c",
    "mixed_fallback_backward": "34c67c29811a548264dbc00847d3eb056d5d159e5c5e4d27398538223793d24c",
    "scaling": "fca3091e0b8446c0075eef531b766e63e0031dcea16914a66eb47d33cb391a3e",
}
THEOREM_SCOPE = (
    "bounded event-certified projective chart segments; "
    "no 360/720 multi-chart claim"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "research_notes"
    / "gauged_uvt_trace_atlas"
    / "paper"
    / "generated"
    / "schema_v2"
)
DEFAULT_PAPER_DRAFT = (
    ROOT
    / "research_notes"
    / "gauged_uvt_trace_atlas"
    / "paper"
    / "WORLD_TUBES_PAPER_DRAFT.md"
)
DEFAULT_PAPER_TEX = DEFAULT_PAPER_DRAFT.with_name("WORLD_TUBES_PAPER.tex")
DEFAULT_PAPER_BIBLIOGRAPHY = DEFAULT_PAPER_DRAFT.with_name(
    "WORLD_TUBES_REFERENCES.bib"
)
MANUSCRIPT_TABLE_INPUTS = tuple(
    (
        "research_notes/gauged_uvt_trace_atlas/paper/generated/"
        f"schema_v2/{name}"
    )
    for name in (
        "theorem_table.tex",
        "frozen_scaling_table.tex",
        "variable_camera_table.tex",
        "public_context_table.tex",
    )
)
FORBIDDEN_MANUSCRIPT_EVIDENCE = (
    "5.9153",
    "5.6159",
    "4.9110",
    "coffee_progressive_heldout_psnr",
    "0.047677",
    "0.181323",
    "0.392235",
    "Bounded-orbit compiled forward is faster",
    "Bounded-orbit compiled backward is faster",
)


@dataclass(frozen=True)
class ExpectedRun:
    ordinal: int
    role: str
    protocol_path: Path
    protocol_name: str
    seed: int
    backward_policy: str
    worldfoam_initializer: str

    @property
    def key(self) -> str:
        return (
            f"{self.protocol_name}/seed_{self.seed}/"
            f"{self.backward_policy}"
        )

    def summary_path(self, run_root: Path) -> Path:
        return (
            run_root
            / self.protocol_name
            / f"seed_{self.seed}"
            / "run_summary.json"
        )

    def as_dict(self, run_root: Path) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "key": self.key,
            "role": self.role,
            "protocol": _display_path(self.protocol_path),
            "protocol_name": self.protocol_name,
            "seed": self.seed,
            "world_tubes_backward_policy": self.backward_policy,
            "worldfoam_initializer": self.worldfoam_initializer,
            "expected_summary": _display_path(self.summary_path(run_root)),
        }

    def matrix_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "role": self.role,
            "protocol": _display_path(self.protocol_path),
            "seed": self.seed,
            "world_tubes_backward_policy": self.backward_policy,
            "worldfoam_initializer": self.worldfoam_initializer,
        }


def resolve_matrix_run_root(
    matrix_path: Path,
    requested_run_root: Path | None,
) -> Path:
    if requested_run_root is not None:
        candidate = requested_run_root.expanduser()
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (ROOT / candidate).resolve()
        )
    matrix = load_config_file(matrix_path)
    output_root = matrix.get("output_root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ValueError(
            "paper matrix output_root must be a nonempty path when "
            "--run-root is omitted"
        )
    candidate = Path(output_root).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (ROOT / candidate).resolve()
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


def _require_numeric_section(
    errors: list[str],
    section_name: str,
    section: Any,
    keys: Sequence[str],
) -> None:
    if not isinstance(section, Mapping):
        errors.append(f"{section_name} must be an object")
        return
    for key in keys:
        if key not in section:
            errors.append(f"{section_name}.{key} is missing")
        elif not _finite_number(section[key]):
            errors.append(f"{section_name}.{key} must be finite numeric")


def load_expected_runs(matrix_path: Path) -> tuple[str, list[ExpectedRun]]:
    matrix = load_config_file(matrix_path)
    matrix_name = matrix.get("name")
    raw_runs = matrix.get("runs")
    if not isinstance(matrix_name, str) or not matrix_name:
        raise ValueError("paper matrix must have a non-empty name")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("paper matrix must contain at least one run")
    expected: list[ExpectedRun] = []
    ordinal = 0
    for index, raw in enumerate(raw_runs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"matrix run {index} must be an object")
        protocol_value = raw.get("protocol")
        seeds = raw.get("seeds")
        role = raw.get("role")
        policy = raw.get("world_tubes_backward_policy")
        if not isinstance(protocol_value, str):
            raise ValueError(f"matrix run {index} protocol must be a string")
        protocol_path = Path(protocol_value)
        if not protocol_path.is_absolute():
            protocol_path = ROOT / protocol_path
        if not protocol_path.is_file():
            raise FileNotFoundError(protocol_path)
        protocol = load_config_file(protocol_path)
        protocol_name = protocol.get("name")
        if not isinstance(protocol_name, str) or not protocol_name:
            raise ValueError(f"protocol {protocol_path} has no name")
        if not isinstance(role, str) or not role:
            raise ValueError(f"matrix run {index} has no role")
        if not isinstance(policy, str) or not policy:
            raise ValueError(f"matrix run {index} has no backward policy")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        ):
            raise ValueError(f"matrix run {index} seeds must be integers")
        for seed in seeds:
            expected.append(
                ExpectedRun(
                    ordinal=ordinal,
                    role=role,
                    protocol_path=protocol_path.resolve(),
                    protocol_name=protocol_name,
                    seed=int(seed),
                    backward_policy=policy,
                    worldfoam_initializer=str(
                        raw.get(
                            "worldfoam_initializer",
                            DEFAULT_WORLDFOAM_INITIALIZER,
                        )
                    ),
                )
            )
            ordinal += 1
    keys = [run.key for run in expected]
    if len(keys) != len(set(keys)):
        raise ValueError("paper matrix expands to duplicate run keys")
    output_keys = [(run.protocol_name, run.seed) for run in expected]
    if len(output_keys) != len(set(output_keys)):
        raise ValueError("paper matrix expands to colliding summary paths")
    return matrix_name, expected


def _validate_source(summary: Mapping[str, Any], errors: list[str]) -> None:
    source = summary.get("source")
    if not isinstance(source, Mapping):
        errors.append("source provenance is missing")
        return
    if summary.get("source_finish") != source:
        errors.append("source_finish does not match source")
    if source.get("repository_dirty") is not False:
        errors.append("repository source is dirty")
    if source.get("star_uvt_dirty") is not False:
        errors.append("STAR UVT source is dirty")
    for key in ("repository_commit", "star_uvt_commit"):
        if not _is_commit(source.get(key)):
            errors.append(f"source.{key} is not a full commit")


def _validate_common_contract(
    summary: Mapping[str, Any],
    errors: list[str],
) -> None:
    common = summary.get("common_evidence_contract")
    if not isinstance(common, Mapping):
        errors.append("common_evidence_contract is missing")
        return
    if int(common.get("schema_version", -1)) != 1:
        errors.append("common_evidence_contract schema is stale")
    for key in (
        "dataset_input_identity",
        "decoded_dataset_bundle",
        "evaluator",
        "runtime",
        "sample_schedule",
    ):
        contract = common.get(key)
        if not isinstance(contract, Mapping) or not _is_sha256(
            contract.get("sha256")
        ):
            errors.append(f"common_evidence_contract.{key} identity is invalid")
    manifest = summary.get("manifest_validation")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("input_identity") != common.get("dataset_input_identity")
    ):
        errors.append("manifest input identity does not match common evidence")
    if not isinstance(manifest, Mapping):
        return
    dataset_family = manifest.get("dataset")
    try:
        expected_pose_source = expected_paper_pose_source(dataset_family)
    except ValueError as error:
        errors.append(str(error))
        return
    if manifest.get("expected_pose_source") != expected_pose_source:
        errors.append("manifest expected pose source is not canonical")
    input_identity = manifest.get("input_identity")
    if (
        not isinstance(input_identity, Mapping)
        or input_identity.get("dataset") != dataset_family
    ):
        errors.append("manifest dataset family does not match raw input identity")
    decoded_bundle = common.get("decoded_dataset_bundle")
    if (
        not isinstance(decoded_bundle, Mapping)
        or decoded_bundle.get("pose_source") != expected_pose_source
    ):
        errors.append(
            "decoded dataset pose source does not match the validated "
            f"{dataset_family!r} manifest"
        )


def _validate_wandb(
    lane_name: str,
    lane: Mapping[str, Any],
    errors: list[str],
) -> None:
    wandb = lane.get("wandb")
    if not isinstance(wandb, Mapping):
        errors.append(f"{lane_name}.wandb is missing")
        return
    if wandb.get("mode") not in {"online", "offline"}:
        errors.append(f"{lane_name}.wandb mode is invalid")
    if not isinstance(wandb.get("run_id"), str) or not wandb["run_id"].strip():
        errors.append(f"{lane_name}.wandb run_id is missing")
    run_file = wandb.get("run_file")
    if not isinstance(run_file, Mapping) or not _is_sha256(
        run_file.get("sha256")
    ):
        errors.append(f"{lane_name}.wandb finalized run-file identity is missing")


def _validate_lane(
    lane_name: str,
    lane: Any,
    common: Mapping[str, Any],
    errors: list[str],
) -> None:
    if not isinstance(lane, Mapping):
        errors.append(f"{lane_name} lane is missing")
        return
    evidence = lane.get("evidence")
    if not isinstance(evidence, Mapping):
        errors.append(f"{lane_name}.evidence is missing")
        return
    if int(evidence.get("schema_version", -1)) != PAPER_EVIDENCE_SCHEMA_VERSION:
        errors.append(
            f"{lane_name}.evidence must use schema "
            f"{PAPER_EVIDENCE_SCHEMA_VERSION}"
        )
    _require_numeric_section(
        errors,
        f"{lane_name}.evidence.quality",
        evidence.get("quality"),
        QUALITY_KEYS,
    )
    _require_numeric_section(
        errors,
        f"{lane_name}.evidence.cost",
        evidence.get("cost"),
        COST_KEYS,
    )
    _require_numeric_section(
        errors,
        f"{lane_name}.evidence.timing",
        evidence.get("timing"),
        TIMING_KEYS,
    )
    quality = evidence.get("quality")
    if isinstance(quality, Mapping):
        for key in ("eval_l1", "heldout_eval_l1", "heldout_eval_lpips"):
            if _finite_number(quality.get(key)) and float(quality[key]) < 0.0:
                errors.append(f"{lane_name}.evidence.quality.{key} is negative")
        for key in ("eval_ssim", "heldout_eval_ssim"):
            if _finite_number(quality.get(key)) and not (
                -1.0 <= float(quality[key]) <= 1.0
            ):
                errors.append(f"{lane_name}.evidence.quality.{key} is outside [-1,1]")
    cost = evidence.get("cost")
    if isinstance(cost, Mapping):
        for key in (
            "optimizer_steps",
            "target_frames",
            "rasterized_frames",
            "target_pixels",
            "rasterized_pixels",
            "parameter_count",
            "trainable_parameter_count",
            "parameter_bytes",
            "serialized_checkpoint_bytes",
        ):
            if _finite_number(cost.get(key)) and float(cost[key]) <= 0.0:
                errors.append(f"{lane_name}.evidence.cost.{key} must be positive")
    if not isinstance(evidence.get("diagnostics"), Mapping) or not evidence[
        "diagnostics"
    ]:
        errors.append(f"{lane_name}.evidence.diagnostics is missing")
    route_native = lane.get("route_native_extension")
    if not isinstance(route_native, Mapping) or not _is_sha256(
        route_native.get("sha256")
    ):
        errors.append(f"{lane_name}.route_native_extension identity is missing")
    paper = lane.get("paper_protocol")
    if not isinstance(paper, Mapping):
        errors.append(f"{lane_name}.paper_protocol is missing")
    else:
        for paper_key, common_key in (
            ("paper_dataset_bundle", "decoded_dataset_bundle"),
            ("paper_evaluator", "evaluator"),
            ("paper_runtime", "runtime"),
            ("sample_schedule", "sample_schedule"),
        ):
            if paper.get(paper_key) != common.get(common_key):
                errors.append(
                    f"{lane_name}.paper_protocol.{paper_key} does not match "
                    "the common evidence contract"
                )
    _validate_wandb(lane_name, lane, errors)


def validate_run_summary(
    expected: ExpectedRun,
    summary: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if summary.get("status") != "complete":
        errors.append("status must be complete")
    if int(summary.get("seed", -1)) != expected.seed:
        errors.append("seed does not match the matrix slot")
    protocol = summary.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("name") != expected.protocol_name
    ):
        errors.append("protocol does not match the matrix slot")
    requested_policy = summary.get(
        "world_tubes_requested_backward_policy",
        summary.get("world_tubes_backward_policy"),
    )
    if requested_policy != expected.backward_policy:
        errors.append("requested backward policy does not match the matrix slot")
    _validate_source(summary, errors)
    _validate_common_contract(summary, errors)
    common = summary.get("common_evidence_contract")
    if not isinstance(common, Mapping):
        common = {}
    lanes = summary.get("lanes")
    if not isinstance(lanes, Mapping):
        errors.append("lanes are missing")
        return errors
    if set(lanes) != set(LANE_ORDER):
        errors.append("lanes must contain exactly World Tubes, WorldFoam, and dynamic 3DGS")
    for lane_name in LANE_ORDER:
        _validate_lane(lane_name, lanes.get(lane_name), common, errors)
    valid_costs = [
        lanes[lane_name]["evidence"]["cost"]
        for lane_name in LANE_ORDER
        if isinstance(lanes.get(lane_name), Mapping)
        and isinstance(lanes[lane_name].get("evidence"), Mapping)
        and isinstance(lanes[lane_name]["evidence"].get("cost"), Mapping)
    ]
    for key in (
        "optimizer_steps",
        "target_frames",
        "rasterized_frames",
        "target_pixels",
        "rasterized_pixels",
    ):
        values = {
            float(cost[key])
            for cost in valid_costs
            if _finite_number(cost.get(key))
        }
        if len(values) != 1:
            errors.append(f"lane cost {key} is not matched")
    for cost in valid_costs:
        if (
            _finite_number(cost.get("target_frames"))
            and _finite_number(cost.get("rasterized_frames"))
            and float(cost["target_frames"]) != float(cost["rasterized_frames"])
        ):
            errors.append("target and rasterized frame counts differ")
        if (
            _finite_number(cost.get("target_pixels"))
            and _finite_number(cost.get("rasterized_pixels"))
            and float(cost["target_pixels"]) != float(cost["rasterized_pixels"])
        ):
            errors.append("target and rasterized pixel counts differ")
    return sorted(set(errors))


def _validate_public_run_deep(
    expected: ExpectedRun,
    summary: Mapping[str, Any],
    summary_path: Path,
) -> list[str]:
    """Reuse the matrix runner's retained-artifact validator verbatim."""

    run = matrix_runner.MatrixRun(
        role=expected.role,
        protocol_path=expected.protocol_path,
        seed=expected.seed,
        backward_policy=expected.backward_policy,
        worldfoam_initializer=expected.worldfoam_initializer,
    )
    try:
        protocol = matrix_runner.resolve_paper_training_protocol(
            load_config_file(expected.protocol_path)
        )
        matrix_runner.validate_existing_summary(
            run,
            summary,
            protocol=protocol,
            summary_path=summary_path,
        )
    except Exception as error:
        return [f"matrix-run deep validation failed: {error}"]
    return []


def _flatten_run(
    expected: ExpectedRun,
    summary: Mapping[str, Any],
    summary_path: Path,
) -> list[dict[str, Any]]:
    protocol = summary["protocol"]
    common = summary["common_evidence_contract"]
    source = summary["source"]
    rows: list[dict[str, Any]] = []
    for lane_name in LANE_ORDER:
        lane = summary["lanes"][lane_name]
        evidence = lane["evidence"]
        rows.append(
            {
                "matrix_ordinal": expected.ordinal,
                "run_key": expected.key,
                "role": expected.role,
                "protocol": expected.protocol_name,
                "scene_sample": protocol["dataset"]["sample_id"],
                "train_cameras": "+".join(protocol["dataset"]["train_cameras"]),
                "heldout_cameras": "+".join(
                    protocol["dataset"]["heldout_cameras"]
                ),
                "seed": expected.seed,
                "lane": lane_name,
                "backward_policy": (
                    expected.backward_policy
                    if lane_name == "world_tubes"
                    else "n/a"
                ),
                "repository_commit": source["repository_commit"],
                "star_uvt_commit": source["star_uvt_commit"],
                "dataset_input_sha256": common["dataset_input_identity"]["sha256"],
                "decoded_dataset_sha256": common["decoded_dataset_bundle"]["sha256"],
                "evaluator_sha256": common["evaluator"]["sha256"],
                "runtime_sha256": common["runtime"]["sha256"],
                "sample_schedule_sha256": common["sample_schedule"]["sha256"],
                "route_native_sha256": lane["route_native_extension"]["sha256"],
                "wandb_run_id": lane["wandb"]["run_id"],
                "run_summary": _display_path(summary_path),
                **evidence["quality"],
                **evidence["cost"],
                **evidence["timing"],
                "diagnostics_json": json.dumps(
                    evidence["diagnostics"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return rows


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    group_order: list[tuple[int, str]] = []
    for row in rows:
        key = (int(row["matrix_ordinal"]), str(row["lane"]))
        # Repeats of one protocol/role occupy consecutive matrix ordinals.  Use
        # the first ordinal of the protocol/role/lane as the stable group id.
        compatible_key = next(
            (
                existing
                for existing in group_order
                if existing[1] == key[1]
                and groups[existing][0]["role"] == row["role"]
                and groups[existing][0]["protocol"] == row["protocol"]
            ),
            None,
        )
        if compatible_key is None:
            compatible_key = key
            group_order.append(compatible_key)
            groups[compatible_key] = []
        groups[compatible_key].append(row)
    aggregated: list[dict[str, Any]] = []
    for key in group_order:
        group = groups[key]
        first = group[0]
        seeds = [int(row["seed"]) for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                f"duplicate seeds in {first['role']}/{first['protocol']}/"
                f"{first['lane']}"
            )
        compatibility = (
            "repository_commit",
            "star_uvt_commit",
            "dataset_input_sha256",
            "decoded_dataset_sha256",
            "evaluator_sha256",
            "runtime_sha256",
            "backward_policy",
            "route_native_sha256",
        )
        drift = [
            field
            for field in compatibility
            if len({str(row[field]) for row in group}) != 1
        ]
        if drift:
            raise ValueError(
                f"incompatible repeats for {first['role']}/"
                f"{first['protocol']}/{first['lane']}: {', '.join(drift)}"
            )
        result: dict[str, Any] = {
            "ordinal": min(int(row["matrix_ordinal"]) for row in group),
            "role": first["role"],
            "protocol": first["protocol"],
            "lane": first["lane"],
            "seeds": seeds,
            "repeat_count": len(group),
        }
        for metric in AGGREGATE_METRIC_KEYS:
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = statistics.fmean(values)
            result[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregated.append(result)
    return aggregated


def _resolve_recorded_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _canonical_matrix_bindings(
    *,
    matrix_name: str,
    expected_runs: Sequence[ExpectedRun],
    run_root: Path,
    matrix_summary_path: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[str]], list[str]]:
    """Bind runner-validated matrix records to exact retained summaries."""

    if not matrix_summary_path.is_file():
        return {}, {}, ["canonical matrix_summary.json is missing"]
    try:
        matrix_summary = _load_json(matrix_summary_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {}, {}, [f"could not load canonical matrix summary: {error}"]

    errors: list[str] = []
    bindings: dict[str, Mapping[str, Any]] = {}
    binding_errors: dict[str, list[str]] = {}
    expected_count = len(expected_runs)
    if matrix_summary.get("status") != "complete":
        errors.append("canonical matrix status must be complete")
    if matrix_summary.get("matrix") != matrix_name:
        errors.append("canonical matrix name does not match the matrix config")
    if int(matrix_summary.get("run_count", -1)) != expected_count:
        errors.append("canonical matrix run_count is inconsistent")
    if int(matrix_summary.get("lane_row_count", -1)) != (
        expected_count * len(LANE_ORDER)
    ):
        errors.append("canonical matrix lane_row_count is inconsistent")
    records = matrix_summary.get("runs")
    if not isinstance(records, list):
        errors.append("canonical matrix runs must be a list")
        records = []
    if len(records) != expected_count:
        errors.append("canonical matrix does not contain every expected run")

    actual_keys = [
        record.get("run", {}).get("key")
        for record in records
        if isinstance(record, Mapping)
        and isinstance(record.get("run"), Mapping)
    ]
    expected_keys = [expected.key for expected in expected_runs]
    if actual_keys != expected_keys:
        errors.append(
            "canonical matrix run keys/order do not exactly match the matrix config"
        )

    for index, expected in enumerate(expected_runs):
        run_errors: list[str] = []
        if index >= len(records) or not isinstance(records[index], Mapping):
            run_errors.append("canonical matrix record is missing")
            binding_errors[expected.key] = run_errors
            continue
        record = records[index]
        recorded_run = record.get("run")
        if recorded_run != expected.matrix_record():
            run_errors.append("canonical matrix run contract does not match")
        embedded = record.get("summary")
        if not isinstance(embedded, Mapping):
            run_errors.append("canonical matrix embedded summary is missing")
            binding_errors[expected.key] = run_errors
            continue
        summary_path = expected.summary_path(run_root)
        embedded_path = _resolve_recorded_path(
            embedded.get("run_summary_path")
        )
        if embedded_path != summary_path.resolve():
            run_errors.append(
                "canonical matrix run_summary_path does not bind the retained summary"
            )
        if not summary_path.is_file():
            run_errors.append("canonical matrix retained run_summary.json is missing")
        else:
            try:
                retained = _load_json(summary_path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                run_errors.append(f"could not load retained summary: {error}")
            else:
                embedded_without_path = dict(embedded)
                embedded_without_path.pop("run_summary_path", None)
                if embedded_without_path != retained:
                    run_errors.append(
                        "canonical matrix embedded summary does not exactly "
                        "match retained run_summary.json"
                    )
                else:
                    bindings[expected.key] = retained
        if run_errors:
            binding_errors[expected.key] = run_errors
    return bindings, binding_errors, sorted(set(errors))


def collect_public_evidence(
    matrix_path: Path,
    run_root: Path,
    matrix_summary_path: Path,
) -> dict[str, Any]:
    matrix_name, expected_runs = load_expected_runs(matrix_path)
    bindings, binding_errors, matrix_errors = _canonical_matrix_bindings(
        matrix_name=matrix_name,
        expected_runs=expected_runs,
        run_root=run_root,
        matrix_summary_path=matrix_summary_path,
    )
    slots: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for expected in expected_runs:
        summary_path = expected.summary_path(run_root)
        slot = expected.as_dict(run_root)
        if not summary_path.is_file():
            errors = binding_errors.get(expected.key, [])
            slots.append(
                {
                    **slot,
                    "status": "missing" if not errors else "rejected",
                    "errors": errors,
                }
            )
            continue
        if matrix_errors or expected.key in binding_errors:
            slots.append(
                {
                    **slot,
                    "status": "rejected",
                    "errors": [
                        *matrix_errors,
                        *binding_errors.get(expected.key, []),
                    ],
                }
            )
            continue
        summary = bindings.get(expected.key)
        if summary is None:
            slots.append(
                {
                    **slot,
                    "status": "rejected",
                    "errors": ["canonical matrix summary binding is absent"],
                }
            )
            continue
        errors = [
            *_validate_public_run_deep(expected, summary, summary_path),
            *validate_run_summary(expected, summary),
        ]
        if errors:
            slots.append(
                {
                    **slot,
                    "status": "rejected",
                    "errors": sorted(set(errors)),
                }
            )
            continue
        summary_sha256 = _file_sha256(summary_path)
        slots.append(
            {
                **slot,
                "status": "accepted",
                "errors": [],
                "summary_sha256": summary_sha256,
            }
        )
        rows.extend(_flatten_run(expected, summary, summary_path))
    accepted = [slot for slot in slots if slot["status"] == "accepted"]
    missing = [slot for slot in slots if slot["status"] == "missing"]
    rejected = [slot for slot in slots if slot["status"] == "rejected"]
    complete = (
        not matrix_errors
        and not binding_errors
        and len(accepted) == len(slots)
        and not rejected
    )
    aggregated = _aggregate_rows(rows) if complete else []
    return {
        "status": (
            "accepted"
            if complete
            else ("invalid" if rejected else "missing")
        ),
        "accepted": complete,
        "matrix": matrix_name,
        "matrix_path": _display_path(matrix_path),
        "matrix_sha256": _file_sha256(matrix_path),
        "matrix_summary": _display_path(matrix_summary_path),
        "matrix_summary_sha256": (
            _file_sha256(matrix_summary_path)
            if matrix_summary_path.is_file()
            else None
        ),
        "matrix_summary_errors": matrix_errors,
        "run_root": _display_path(run_root),
        "expected_run_count": len(slots),
        "accepted_run_count": len(accepted),
        "missing_run_count": len(missing),
        "rejected_run_count": len(rejected),
        "slots": slots,
        # Partial numeric rows are deliberately withheld.  Accepted slot
        # identities remain visible in the ledger without becoming a table.
        "rows": rows if complete else [],
        "aggregated": aggregated,
    }


def _derive_theorem_rows(
    *,
    source_root: Path,
    errors: list[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    reports: dict[str, Mapping[str, Any]] = {}
    source_sha256: dict[str, str] = {}
    for name, relative_path in THEOREM_SOURCE_PATHS.items():
        source_path = source_root / relative_path
        if not source_path.is_file():
            errors.append(f"theorem source is missing: {relative_path}")
            continue
        digest = _file_sha256(source_path)
        source_sha256[name] = digest
        if digest != THEOREM_SOURCE_SHA256[name]:
            errors.append(f"theorem source hash mismatch: {relative_path}")
            continue
        try:
            report = _load_json(source_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"theorem source {name} is unreadable: {error}")
            continue
        if not isinstance(report, Mapping):
            errors.append(f"theorem source {name} is not an object")
            continue
        reports[name] = report
    if set(reports) != set(THEOREM_SOURCE_PATHS):
        return [], source_sha256
    try:
        visibility_rows = {
            row["case_id"]: row
            for row in reports["visibility"]["rows"]
        }
        scaling_rows = reports["scaling"]["rows"]
        fixed_last = next(
            row
            for row in scaling_rows
            if row["route"] == "fixed_chart" and int(row["frames"]) == 128
        )
        replay_last = next(
            row
            for row in scaling_rows
            if row["route"] == "per_frame" and int(row["frames"]) == 128
        )
        trace_ratio = float(fixed_last["trace_count"]) / float(
            replay_last["trace_count"]
        )
        rows = [
            {
                "claim": "Fiber value is gauge invariant",
                "metric": "max relative error",
                "value": reports["gauge_value"]["summary"]["max_rel_error"],
                "acceptance": "<= 1e-10",
                "source": "gauge_value",
            },
            {
                "claim": "Fiber gradient is gauge invariant",
                "metric": "max gradient relative error",
                "value": reports["gauge_gradient"]["summary"][
                    "max_gradient_rel_error"
                ],
                "acceptance": "<= 1e-9",
                "source": "gauge_gradient",
            },
            {
                "claim": "Compiled atlas matches dense/replay image",
                "metric": "max absolute image error",
                "value": reports["decisive_demo"]["summary"][
                    "max_image_abs_error_vs_reference"
                ],
                "acceptance": "<= 1e-5",
                "source": "decisive_demo",
            },
            {
                "claim": (
                    "Unstratified interval exposes an order-crossing failure"
                ),
                "metric": "raw crossing quality error",
                "value": visibility_rows["crossing_raw_interval"][
                    "quality_error"
                ],
                "acceptance": "> 1e-5 (expected failure)",
                "source": "visibility",
            },
            {
                "claim": "Visibility crossing is repaired by stratification",
                "metric": "stratified crossing quality error",
                "value": visibility_rows["crossing_stratified"][
                    "quality_error"
                ],
                "acceptance": "<= 1e-5",
                "source": "visibility",
            },
            {
                "claim": "Finite exposure / rolling shutter forward parity",
                "metric": "max Metal absolute error",
                "value": reports["exposure"]["summary"][
                    "max_metal_abs_error"
                ],
                "acceptance": "<= 1e-5",
                "source": "exposure",
            },
            {
                "claim": "Finite exposure / rolling shutter gradient parity",
                "metric": "max Metal gradient relative error",
                "value": reports["exposure_backward"]["summary"][
                    "max_metal_grad_rel_error"
                ],
                "acceptance": "<= 1e-5",
                "source": "exposure_backward",
            },
            {
                "claim": "Mixed fallback preserves gradients",
                "metric": "max mixed gradient relative error",
                "value": reports["mixed_fallback_backward"]["summary"][
                    "max_mixed_grad_rel_error"
                ],
                "acceptance": "<= 1e-5",
                "source": "mixed_fallback_backward",
            },
            {
                "claim": "Bounded-orbit chart reuses trace state at F=128",
                "metric": "fixed/per-frame trace-count ratio",
                "value": trace_ratio,
                "acceptance": "< 0.25",
                "source": "scaling",
            },
        ]
    except (KeyError, StopIteration, TypeError, ValueError, ZeroDivisionError) as error:
        errors.append(f"cannot rederive theorem rows: {error}")
        return [], source_sha256
    if tuple(reports["scaling"].get("frame_counts", ())) != (
        4,
        8,
        16,
        32,
        64,
        128,
    ):
        errors.append("theorem scaling source frame counts drifted")
    return rows, source_sha256


def collect_theorem_evidence(
    path: Path,
    *,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "missing",
            "accepted": False,
            "input": _display_path(path),
            "errors": [],
            "rows": [],
        }
    try:
        report = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "invalid",
            "accepted": False,
            "input": _display_path(path),
            "errors": [str(error)],
            "rows": [],
        }
    errors: list[str] = []
    rows = report.get("rows")
    summary = report.get("summary")
    expected_rows, source_sha256 = _derive_theorem_rows(
        source_root=source_root,
        errors=errors,
    )
    if report.get("status") != "complete":
        errors.append("status must be complete")
    if report.get("scope") != THEOREM_SCOPE:
        errors.append("theorem scope is not canonical")
    if report.get("sources") != THEOREM_SOURCE_PATHS:
        errors.append("theorem source paths are not canonical")
    if report.get("source_sha256") != THEOREM_SOURCE_SHA256:
        errors.append("theorem source digests are not canonical")
    if not isinstance(summary, Mapping) or summary.get(
        "all_sources_verified"
    ) is not True:
        errors.append("all theorem sources must be verifier-accepted")
    if (
        not isinstance(summary, Mapping)
        or summary.get("timing_claims_excluded") is not True
    ):
        errors.append(
            "bounded-fixture timing claims must be excluded from theorem evidence"
        )
    if not isinstance(rows, list) or not rows:
        errors.append("theorem rows are missing")
        rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"theorem row {index} is invalid")
            continue
        if not _finite_number(row.get("value")):
            errors.append(f"theorem row {index} value is not finite")
        for key in ("claim", "metric", "acceptance", "source"):
            if not isinstance(row.get(key), str) or not row[key]:
                errors.append(f"theorem row {index} {key} is missing")
    if isinstance(summary, Mapping) and int(summary.get("row_count", -1)) != len(
        rows
    ):
        errors.append("theorem row count is inconsistent")
    if rows != expected_rows:
        errors.append(
            "theorem rows do not exactly match pinned retained source reports"
        )
    return {
        "status": "accepted" if not errors else "invalid",
        "accepted": not errors,
        "input": _display_path(path),
        "input_sha256": _file_sha256(path),
        "errors": sorted(set(errors)),
        "scope": report.get("scope"),
        "sources": report.get("sources", {}),
        "source_sha256": source_sha256,
        "rows": list(expected_rows) if not errors else [],
    }


def _timing_quantile(samples: Sequence[float], probability: float) -> float:
    position = float(len(samples) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(samples[lower])
    fraction = position - float(lower)
    return float(samples[lower]) * (1.0 - fraction) + float(
        samples[upper]
    ) * fraction


def _timing_summary(samples: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in samples)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": _timing_quantile(ordered, 0.25),
        "median": statistics.median(ordered),
        "p75": _timing_quantile(ordered, 0.75),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _timing_values_match(actual: Any, expected: float | int) -> bool:
    if isinstance(expected, int):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, int)
            and actual == expected
        )
    return _finite_number(actual) and math.isclose(
        float(actual),
        float(expected),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )


def _validate_frozen_timing(
    *,
    row: Mapping[str, Any],
    row_index: int,
    expected_warmups: int,
    expected_repeats: int,
    errors: list[str],
) -> None:
    label = f"frozen row {row_index}"
    frame_count = int(row.get("frame_count", 0))
    if frame_count < 1:
        errors.append(f"{label} frame count is invalid for timing")
        return
    timing = row.get("timing_benchmark")
    if not isinstance(timing, Mapping):
        errors.append(f"{label} publication timing is missing")
        return
    if (
        int(timing.get("schema_version", -1)) != 1
        or timing.get("status") != "complete"
        or timing.get("publication_ready") is not True
        or timing.get("label") != "warmed_repeated_wall_timing_v1"
        or int(timing.get("warmups", -1)) != expected_warmups
        or int(timing.get("repeats", 0)) != expected_repeats
    ):
        errors.append(f"{label} publication timing contract is invalid")
    samples = timing.get("samples_s")
    summaries = timing.get("summary_s")
    if not isinstance(samples, Mapping) or set(samples) != set(
        PUBLICATION_TIMING_KEYS
    ):
        errors.append(f"{label} timing samples have missing or extra keys")
        return
    if not isinstance(summaries, Mapping) or set(summaries) != set(
        PUBLICATION_TIMING_KEYS
    ):
        errors.append(f"{label} timing summaries have missing or extra keys")
        return
    expected_summary_keys = {
        "count",
        "min",
        "p25",
        "median",
        "p75",
        "max",
        "mean",
    }
    for key in PUBLICATION_TIMING_KEYS:
        values = samples[key]
        if (
            not isinstance(values, list)
            or len(values) != expected_repeats
            or any(
                not _finite_number(value) or float(value) < 0.0
                for value in values
            )
        ):
            errors.append(
                f"{label} timing samples {key} must contain exactly "
                f"{expected_repeats} finite nonnegative values"
            )
            continue
        reported_summary = summaries[key]
        if not isinstance(reported_summary, Mapping) or set(
            reported_summary
        ) != expected_summary_keys:
            errors.append(f"{label} timing summary {key} has an invalid shape")
            continue
        recomputed = _timing_summary([float(value) for value in values])
        for statistic, expected_value in recomputed.items():
            if not _timing_values_match(
                reported_summary.get(statistic),
                expected_value,
            ):
                errors.append(
                    f"{label} timing summary {key}.{statistic} "
                    "does not match raw samples"
                )
    if any(
        not isinstance(samples[key], list)
        or len(samples[key]) != expected_repeats
        or any(not _finite_number(value) for value in samples[key])
        for key in PUBLICATION_TIMING_KEYS
    ):
        return
    for sample_index in range(expected_repeats):
        replay_forward = float(samples["replay_total_forward"][sample_index])
        replay_backward = float(samples["replay_total_backward"][sample_index])
        compiled_compile = float(
            samples["compiled_atlas_compile"][sample_index]
        )
        compiled_forward = float(
            samples["compiled_total_forward"][sample_index]
        )
        compiled_backward = float(
            samples["compiled_total_backward"][sample_index]
        )
        expected_derived = {
            "replay_total_forward_backward": (
                replay_forward + replay_backward
            ),
            "replay_per_frame_forward": replay_forward / float(frame_count),
            "replay_per_frame_backward": replay_backward / float(frame_count),
            "compiled_total_forward_backward": (
                compiled_forward + compiled_backward
            ),
            "compiled_compile_plus_forward_backward": (
                compiled_compile + compiled_forward + compiled_backward
            ),
            "compiled_per_frame_forward": (
                compiled_forward / float(frame_count)
            ),
            "compiled_per_frame_backward": (
                compiled_backward / float(frame_count)
            ),
        }
        for key, expected_value in expected_derived.items():
            if not math.isclose(
                float(samples[key][sample_index]),
                expected_value,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                errors.append(
                    f"{label} timing sample {key}[{sample_index}] "
                    "violates its derived identity"
                )


def _validate_frozen_artifact_bindings(
    summary_path: Path,
    summary: Mapping[str, Any],
) -> list[str]:
    """Reopen and deeply verify the learned-world artifacts and current inputs."""

    errors: list[str] = []
    protocol_path = _resolve_recorded_path(summary.get("protocol_path"))
    expected_protocol_sha256 = summary.get("protocol_sha256")
    if (
        protocol_path is None
        or not protocol_path.is_file()
        or not _is_sha256(expected_protocol_sha256)
        or _file_sha256(protocol_path) != expected_protocol_sha256
    ):
        return ["frozen protocol path/hash binding is missing or stale"]

    expected_paths = {
        "comparison_report": summary_path.parent / "comparison_report.json",
        "execution_identity": summary_path.parent / "execution_identity.json",
    }
    resolved_paths: dict[str, Path] = {}
    for key, expected_path in expected_paths.items():
        recorded_path = _resolve_recorded_path(summary.get(key))
        if recorded_path != expected_path.resolve() or not expected_path.is_file():
            errors.append(
                f"frozen {key}.json is not bound to the summary sibling"
            )
            continue
        reported_sha256 = summary.get(f"{key}_sha256")
        if (
            not _is_sha256(reported_sha256)
            or reported_sha256 != _file_sha256(expected_path)
        ):
            errors.append(f"frozen {key}.json hash binding drifted")
            continue
        resolved_paths[key] = expected_path.resolve()
    if errors:
        return sorted(set(errors))

    try:
        protocol = matrix_runner.resolve_paper_training_protocol(
            load_config_file(protocol_path)
        )
        comparison_report = _load_json(resolved_paths["comparison_report"])
        execution_identity = _load_json(resolved_paths["execution_identity"])
        sweep = comparison_report["star_uvt"][
            "frozen_world_replay_compiled_sweep"
        ]
        requested_frame_counts = tuple(
            int(value) for value in sweep["requested_frame_counts"]
        )
        max_frames = int(sweep["primary_requested_frame_count"])
        timing_warmups = int(summary["timing_warmups"])
        timing_repeats = int(summary["timing_repeats"])
        seed = int(summary["seed"])
        frozen_runner.validate_report_identity(
            comparison_report,
            protocol=protocol,
            seed=seed,
            max_frames=max_frames,
            frame_counts=requested_frame_counts,
            timing_warmups=timing_warmups,
            timing_repeats=timing_repeats,
        )

        command = execution_identity.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(value, str) for value in command)
        ):
            raise ValueError("frozen execution command is invalid")
        python = Path(command[0])
        if not python.is_absolute() or not python.is_file():
            raise ValueError("frozen execution Python identity is invalid")
        device = str(comparison_report["meta"]["device"])
        expected_command = frozen_runner.build_command(
            protocol_path,
            protocol,
            seed=seed,
            out_dir=summary_path.parent,
            device=device,
            max_frames=max_frames,
            allow_local_mps_execution=device.lower() == "mps",
            frame_counts=requested_frame_counts,
            timing_warmups=timing_warmups,
            timing_repeats=timing_repeats,
        )
        expected_command[0] = command[0]
        if command != expected_command:
            raise ValueError("frozen execution command drifted")

        current_source = single_runner.source_provenance()
        current_manifest = single_runner.validate_manifest(protocol)
        frozen_runner.validate_execution_identity(
            execution_identity,
            protocol_path=protocol_path,
            command=expected_command,
            report_path=resolved_paths["comparison_report"],
            expected_source=current_source,
            expected_native_extension=comparison_report["meta"][
                "star_uvt_native_extension"
            ],
            expected_dataset_input_identity=current_manifest[
                "input_identity"
            ],
            expected_protocol=protocol.as_dict(),
        )

        expected_common = {
            "schema_version": 1,
            "dataset_input_identity": current_manifest["input_identity"],
            "decoded_dataset_bundle": comparison_report["meta"][
                "paper_dataset_bundle"
            ],
            "evaluator": comparison_report["meta"]["paper_evaluator"],
            "runtime": comparison_report["meta"]["paper_runtime"],
        }
        report_frozen = comparison_report["star_uvt"][
            "frozen_world_replay_compiled"
        ]
        if summary.get("protocol") != protocol.as_dict():
            raise ValueError("frozen wrapper protocol drifted")
        if summary.get("manifest_validation") != current_manifest:
            raise ValueError("frozen wrapper raw dataset identity drifted")
        if summary.get("common_evidence_contract") != expected_common:
            raise ValueError("frozen wrapper common evidence contract drifted")
        if (
            summary.get("source") != execution_identity["source_start"]
            or summary.get("source_finish")
            != execution_identity["source_finish"]
        ):
            raise ValueError("frozen wrapper source identity drifted")
        if summary.get("frozen_world_replay_compiled") != report_frozen:
            raise ValueError("frozen wrapper primary evidence drifted")
        if summary.get("frozen_world_replay_compiled_sweep") != sweep:
            raise ValueError("frozen wrapper sweep evidence drifted")
        resolved = frozen_runner.resolve_frame_counts(
            full_frames=protocol.dataset.frame_count,
            max_frames=max_frames,
            frame_counts=requested_frame_counts,
        )
        if tuple(summary.get("resolved_frame_counts", ())) != resolved:
            raise ValueError("frozen wrapper exact sample-count grid drifted")
    except Exception as error:
        errors.append(f"frozen learned-world deep validation failed: {error}")
    return sorted(set(errors))


def collect_frozen_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "missing",
            "accepted": False,
            "input": _display_path(path),
            "errors": [],
            "rows": [],
        }
    try:
        report = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "invalid",
            "accepted": False,
            "input": _display_path(path),
            "errors": [str(error)],
            "rows": [],
        }
    errors = _validate_frozen_artifact_bindings(path, report)
    if int(report.get("schema_version", -1)) != 1:
        errors.append("frozen wrapper schema is stale")
    if report.get("status") != "accepted":
        errors.append("frozen wrapper status must be accepted")
    if report.get("publication_eligible") is not True:
        errors.append("frozen wrapper is not publication eligible")
    timing_warmups = int(report.get("timing_warmups", -1))
    timing_repeats = int(report.get("timing_repeats", 0))
    if timing_warmups < 1:
        errors.append("frozen timing must use at least one warmup")
    if timing_repeats < 3:
        errors.append("frozen timing must use at least three repeats")
    _validate_source(report, errors)
    sweep = report.get("frozen_world_replay_compiled_sweep")
    rows: list[Mapping[str, Any]] = []
    if not isinstance(sweep, Mapping):
        errors.append("frozen sweep is missing")
    else:
        if (
            int(sweep.get("schema_version", -1)) != 1
            or sweep.get("status") != "complete"
        ):
            errors.append("frozen sweep contract is stale or incomplete")
        if (
            int(sweep.get("timing_benchmark_warmups", -1))
            != timing_warmups
            or int(sweep.get("timing_benchmark_repeats", 0))
            != timing_repeats
        ):
            errors.append(
                "frozen sweep timing controls do not match the wrapper"
            )
        for key in (
            "all_rows_accepted",
            "publication_eligible",
            "selected_time_slice_parity_accepted",
            "all_rows_timing_publication_ready",
            "all_rows_storage_publication_ready",
            "all_rows_route_memory_publication_ready",
            "checkpoint_shared_across_rows",
            "world_state_shared_across_rows",
        ):
            if sweep.get(key) is not True:
                errors.append(f"frozen sweep {key} must be true")
        requested = set(sweep.get("requested_frame_counts", ()))
        full_frames = int(sweep.get("full_dataset_frame_count", 0))
        if not set(CANONICAL_FROZEN_FRAME_COUNTS).issubset(requested):
            errors.append("frozen sweep is missing canonical frame counts")
        if 0 not in requested and full_frames not in requested:
            errors.append("frozen sweep is missing the full-frame row")
        raw_rows = sweep.get("rows")
        if not isinstance(raw_rows, list) or not raw_rows:
            errors.append("frozen sweep rows are missing")
        else:
            rows = [row for row in raw_rows if isinstance(row, Mapping)]
            if len(rows) != len(raw_rows):
                errors.append("frozen sweep contains invalid rows")
    seen_frames: set[int] = set()
    for index, row in enumerate(rows):
        frame_count = int(row.get("frame_count", 0))
        if frame_count < 1 or frame_count in seen_frames:
            errors.append(f"frozen row {index} frame count is invalid")
        seen_frames.add(frame_count)
        if (
            int(row.get("schema_version", -1)) != 2
            or row.get("status") != "complete"
            or row.get("accepted") is not True
        ):
            errors.append(f"frozen row {index} is not accepted schema-v2 evidence")
        for section_name, keys in (
            ("image", ("max_abs_error", "mean_abs_error")),
            (
                "gradient",
                (
                    "global_normalized_l2_error",
                    "max_parameter_normalized_l2_error",
                ),
            ),
            (
                "atlas",
                (
                    "trace_count",
                    "cell_count",
                    "interval_trace_entries",
                    "fallback_fraction",
                ),
            ),
            (
                "payload_bytes",
                ("compiled_to_replay_logical_volume_ratio",),
            ),
        ):
            _require_numeric_section(
                errors,
                f"frozen row {index}.{section_name}",
                row.get(section_name),
                keys,
            )
        payload = row.get("payload_bytes")
        retained_storage = row.get("retained_storage_bytes")
        route_memory = row.get("route_memory")
        if (
            not isinstance(payload, Mapping)
            or payload.get("metric_kind") != "logical_work_volume_proxy"
            or payload.get("topology_bytes_included") is not False
            or payload.get("storage_claim_eligible") is not False
            or payload.get("publication_claim_eligible") is not False
        ):
            errors.append(
                f"frozen row {index} logical payload proxy is mislabeled"
            )
        if (
            not isinstance(retained_storage, Mapping)
            or retained_storage.get("topology_bytes_included") is not True
            or retained_storage.get("storage_claim_eligible") is not True
            or retained_storage.get("publication_claim_eligible") is not True
        ):
            errors.append(
                f"frozen row {index} topology-inclusive storage is missing"
            )
        if (
            not isinstance(route_memory, Mapping)
            or route_memory.get("compiled_parity_replay_excluded") is not True
            or route_memory.get("publication_claim_eligible") is not True
        ):
            errors.append(
                f"frozen row {index} route-scoped memory is missing"
            )
        _validate_frozen_timing(
            row=row,
            row_index=index,
            expected_warmups=timing_warmups,
            expected_repeats=timing_repeats,
            errors=errors,
        )
    required_resolved = {
        *CANONICAL_FROZEN_FRAME_COUNTS,
        int(sweep.get("full_dataset_frame_count", 0))
        if isinstance(sweep, Mapping)
        else 0,
    }
    if rows and not required_resolved.issubset(seen_frames):
        errors.append("frozen sweep resolved rows are incomplete")
    return {
        "status": "accepted" if not errors else "invalid",
        "accepted": not errors,
        "input": _display_path(path),
        "input_sha256": _file_sha256(path),
        "errors": sorted(set(errors)),
        "protocol": report.get("protocol"),
        "seed": report.get("seed"),
        "rows": sorted(
            (dict(row) for row in rows),
            key=lambda row: int(row["frame_count"]),
        )
        if not errors
        else [],
    }


def collect_variable_camera_evidence(
    path: Path,
    *,
    verify_current_source: bool = True,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "missing",
            "accepted": False,
            "input": _display_path(path),
            "errors": [],
            "rows": [],
        }
    try:
        report = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "invalid",
            "accepted": False,
            "input": _display_path(path),
            "errors": [str(error)],
            "rows": [],
        }
    errors = verify_variable_camera_closure_death_curve(report)
    if verify_current_source:
        errors.extend(verify_current_implementation(report))
    summary = report.get("summary")
    rows = report.get("rows")
    if not isinstance(summary, Mapping):
        errors.append("variable-camera summary is missing")
        summary = {}
    acceptance = report.get("acceptance")
    if (
        not isinstance(acceptance, Mapping)
        or acceptance.get("accepted") is not True
        or acceptance.get("label") != "accepted_bounded_closure_death_gate"
    ):
        errors.append("variable-camera closure/death gate is not accepted")
    if not isinstance(rows, list) or not rows:
        errors.append("variable-camera rows are missing")
        rows = []
    return {
        "status": "accepted" if not errors else "invalid",
        "accepted": not errors,
        "input": _display_path(path),
        "input_sha256": _file_sha256(path),
        "errors": sorted(set(errors)),
        "thresholds": report.get("thresholds"),
        "summary": dict(summary),
        "rows": list(rows) if not errors else [],
    }


def build_bundle(
    *,
    matrix_path: Path,
    run_root: Path,
    matrix_summary: Path,
    theorem_summary: Path,
    frozen_summary: Path,
    variable_camera_summary: Path,
    verify_current_variable_camera_source: bool = True,
) -> dict[str, Any]:
    components = {
        "theorem_correctness": collect_theorem_evidence(theorem_summary),
        "public_context": collect_public_evidence(
            matrix_path,
            run_root,
            matrix_summary,
        ),
        "frozen_world_scaling": collect_frozen_evidence(frozen_summary),
        "variable_camera_closure_death": collect_variable_camera_evidence(
            variable_camera_summary,
            verify_current_source=verify_current_variable_camera_source,
        ),
    }
    accepted = all(component["accepted"] is True for component in components.values())
    missing_runtime_inputs: list[dict[str, Any]] = []
    public = components["public_context"]
    if public["matrix_summary_sha256"] is None or public["matrix_summary_errors"]:
        missing_runtime_inputs.append(
            {
                "component": "public_context_matrix_summary",
                "status": (
                    "missing"
                    if public["matrix_summary_sha256"] is None
                    else "invalid"
                ),
                "expected_summary": public["matrix_summary"],
                "validation_errors": public["matrix_summary_errors"],
                "required_contract": {
                    "status": "complete",
                    "matrix": public["matrix"],
                    "run_count": public["expected_run_count"],
                    "lane_row_count": (
                        public["expected_run_count"] * len(LANE_ORDER)
                    ),
                    "exact_ordered_run_keys": [
                        slot["key"] for slot in public["slots"]
                    ],
                    "embedded_summaries_match_retained_json": True,
                },
            }
        )
    for slot in public["slots"]:
        if slot["status"] != "accepted":
            missing_runtime_inputs.append(
                {
                    "component": "public_context",
                    "status": slot["status"],
                    "run_key": slot["key"],
                    "expected_summary": slot["expected_summary"],
                    "protocol": slot["protocol"],
                    "seed": slot["seed"],
                    "world_tubes_backward_policy": slot[
                        "world_tubes_backward_policy"
                    ],
                    "validation_errors": slot["errors"],
                }
            )
    for component_name in (
        "frozen_world_scaling",
        "variable_camera_closure_death",
    ):
        component = components[component_name]
        if component["accepted"] is not True:
            required_contract = (
                {
                    "protocol": (
                        "coffee_martini_full_300f_progressive_512_v1"
                    ),
                    "seed": 17,
                    "max_frames": 0,
                    "frame_counts": [
                        0,
                        *CANONICAL_FROZEN_FRAME_COUNTS,
                    ],
                    "minimum_timing_warmups": 1,
                    "minimum_timing_repeats": 3,
                    "required_gates": [
                        "all_rows_accepted",
                        "publication_eligible",
                        "selected_time_slice_parity_accepted",
                        "all_rows_timing_publication_ready",
                        "all_rows_storage_publication_ready",
                        "all_rows_route_memory_publication_ready",
                        "checkpoint_shared_across_rows",
                        "world_state_shared_across_rows",
                    ],
                }
                if component_name == "frozen_world_scaling"
                else {
                    "benchmark": (
                        "world_tubes_variable_camera_closure_death_curve"
                    ),
                    "schema_version": 1,
                    "required_acceptance_label": (
                        "accepted_bounded_closure_death_gate"
                    ),
                    "required_regimes": ["closure", "death"],
                }
            )
            missing_runtime_inputs.append(
                {
                    "component": component_name,
                    "status": component["status"],
                    "expected_summary": component["input"],
                    "validation_errors": component["errors"],
                    "required_contract": required_contract,
                }
            )
    generator_path = Path(__file__).resolve()
    payload = {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "generator": {
            "name": GENERATOR_NAME,
            "source": _display_path(generator_path),
            "source_sha256": _file_sha256(generator_path),
            "runtime_dependencies": (
                "python_standard_library_plus_paper_runner_deep_validators_"
                "plus_checkpoint_cpu_reader_plus_variable_camera_verifier"
            ),
            "launches_renderer_or_training": False,
            "checkpoint_validation_may_import_torch": True,
        },
        "status": "complete" if accepted else "incomplete",
        "submission_ready": accepted,
        "readiness_scope": "evidence_artifact_bundle_only",
        "manuscript_package_required": True,
        "components": components,
        "missing_runtime_inputs": missing_runtime_inputs,
        "publication_boundary": (
            "Numeric component artifacts are emitted only when that entire "
            "component is verifier-accepted; partial numbers remain absent."
        ),
    }
    payload["ledger_sha256"] = _canonical_json_sha256(payload)
    return payload


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _metric_text(row: Mapping[str, Any], key: str, digits: int) -> str:
    mean = float(row[f"{key}_mean"])
    std = float(row[f"{key}_std"])
    if int(row["repeat_count"]) > 1:
        return f"{mean:.{digits}f} ± {std:.{digits}f}"
    return f"{mean:.{digits}f}"


def _latex_acceptance(value: str) -> str:
    for prefix, operator in (
        ("<= ", r"$\leq$ "),
        (">= ", r"$\geq$ "),
        ("< ", r"$<$ "),
        ("> ", r"$>$ "),
    ):
        if value.startswith(prefix):
            return operator + _latex_escape(value[len(prefix) :])
    return _latex_escape(value)


def _placeholder_markdown(title: str, reason: str) -> str:
    return (
        f"# {title}\n\n"
        f"**NOT SUBMISSION-READY:** {reason}\n\n"
        "No numeric rows were emitted.\n"
    )


def _placeholder_latex(title: str, reason: str) -> str:
    return (
        r"\begin{center}" + "\n"
        + r"\fbox{\begin{minipage}{0.92\linewidth}" + "\n"
        + rf"\textbf{{{_latex_escape(title)} --- NOT SUBMISSION-READY}}\\"
        + "\n"
        + _latex_escape(reason)
        + r" No numeric rows were emitted."
        + "\n"
        + r"\end{minipage}}"
        + "\n"
        + r"\end{center}"
        + "\n"
    )


def _placeholder_svg(title: str, reason: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360" '
        'viewBox="0 0 900 360">\n'
        '<rect width="900" height="360" fill="white"/>\n'
        '<rect x="30" y="30" width="840" height="300" rx="12" '
        'fill="#fff7ed" stroke="#c2410c" stroke-width="3"/>\n'
        f'<text x="450" y="125" text-anchor="middle" font-size="26" '
        f'font-weight="bold">{html.escape(title)}</text>\n'
        '<text x="450" y="175" text-anchor="middle" font-size="20" '
        'fill="#9a3412">NOT SUBMISSION-READY</text>\n'
        f'<text x="450" y="220" text-anchor="middle" font-size="15">'
        f"{html.escape(reason)}</text>\n"
        '<text x="450" y="260" text-anchor="middle" font-size="15">'
        "No numeric data plotted.</text>\n"
        "</svg>\n"
    )


def _write_public_table(
    component: Mapping[str, Any],
    markdown_path: Path,
    latex_path: Path,
) -> None:
    if component["accepted"] is not True:
        reason = (
            f"{component['accepted_run_count']}/"
            f"{component['expected_run_count']} schema-v2 runs accepted."
        )
        markdown_path.write_text(
            _placeholder_markdown("Public representation and cost context", reason),
            encoding="utf-8",
        )
        latex_path.write_text(
            _placeholder_latex("Public representation and cost context", reason),
            encoding="utf-8",
        )
        return
    header = (
        "| Protocol role | Lane | Seeds | PSNR ↑ | SSIM ↑ | LPIPS ↓ | L1 ↓ "
        "| Train wall (s) | Peak driver (GB) | Parameters | Checkpoint (MB) |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    markdown = [
        "# Public representation and cost context",
        "",
        header,
    ]
    latex = [
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        (
            r"Protocol role & Lane & Seeds & PSNR $\uparrow$ & SSIM $\uparrow$ "
            r"& LPIPS $\downarrow$ & L1 $\downarrow$ & Wall (s) "
            r"& Driver (GB) & Params & Checkpoint (MB) \\"
        ),
        r"\midrule",
    ]
    for row in component["aggregated"]:
        seeds = ",".join(str(seed) for seed in row["seeds"])
        cells = (
            _metric_text(row, "heldout_eval_psnr", 3),
            _metric_text(row, "heldout_eval_ssim", 4),
            _metric_text(row, "heldout_eval_lpips", 4),
            _metric_text(row, "heldout_eval_l1", 4),
            _metric_text(row, "train_wall_s", 2),
            f"{float(row['sampled_peak_driver_allocated_bytes_mean']) / 1.0e9:.3f}",
            f"{float(row['parameter_count_mean']):.0f}",
            f"{float(row['serialized_checkpoint_bytes_mean']) / 1.0e6:.3f}",
        )
        markdown.append(
            f"| {row['role']} | {row['lane']} | {seeds} | "
            + " | ".join(cells)
            + " |"
        )
        latex.append(
            f"{_latex_escape(row['role'])} & {_latex_escape(row['lane'])} "
            f"& {seeds} & "
            + " & ".join(
                _latex_escape(cell).replace("±", r"$\pm$")
                for cell in cells
            )
            + r" \\"
        )
    markdown.extend(
        (
            "",
            "Values are mean ± sample standard deviation when multiple seeds "
            "exist; single-seed controls are shown without an uncertainty term.",
        )
    )
    latex.extend((r"\bottomrule", r"\end{tabular}"))
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    latex_path.write_text("\n".join(latex) + "\n", encoding="utf-8")


def _write_theorem_table(
    component: Mapping[str, Any],
    markdown_path: Path,
    latex_path: Path,
) -> None:
    if component["accepted"] is not True:
        reason = "The verifier-produced theorem summary is missing or invalid."
        markdown_path.write_text(
            _placeholder_markdown("Certified correctness", reason),
            encoding="utf-8",
        )
        latex_path.write_text(
            _placeholder_latex("Certified correctness", reason),
            encoding="utf-8",
        )
        return
    markdown = [
        "# Certified correctness",
        "",
        str(component.get("scope", "")),
        "",
        "| Claim | Metric | Value | Acceptance | Source |",
        "|---|---|---:|---:|---|",
    ]
    latex = [
        r"\begin{tabular}{p{0.34\linewidth}p{0.25\linewidth}rr}",
        r"\toprule",
        r"Claim & Metric & Value & Acceptance \\",
        r"\midrule",
    ]
    for row in component["rows"]:
        value = f"{float(row['value']):.6g}"
        markdown.append(
            f"| {row['claim']} | {row['metric']} | {value} | "
            f"{row['acceptance']} | {row['source']} |"
        )
        latex.append(
            f"{_latex_escape(row['claim'])} & {_latex_escape(row['metric'])} "
            f"& {value} & {_latex_acceptance(row['acceptance'])} " + r"\\"
        )
    latex.extend((r"\bottomrule", r"\end{tabular}"))
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    latex_path.write_text("\n".join(latex) + "\n", encoding="utf-8")


def _write_frozen_table(
    component: Mapping[str, Any],
    markdown_path: Path,
    latex_path: Path,
) -> None:
    if component["accepted"] is not True:
        reason = "The frozen identical-world sweep is missing or not publication eligible."
        markdown_path.write_text(
            _placeholder_markdown("Frozen-world compiler scaling", reason),
            encoding="utf-8",
        )
        latex_path.write_text(
            _placeholder_latex("Frozen-world compiler scaling", reason),
            encoding="utf-8",
        )
        return
    markdown = [
        "# Frozen-world compiler scaling",
        "",
        "| F | Image max error | VJP rel. L2 | Fallback | Replay F+B (s) "
        "| Compiled F+B (s) | Compile+F+B (s) | Speedup | Logical-volume ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    latex = [
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        (
            r"$F$ & Image err. & VJP rel. & Fallback & Replay F+B "
            r"& Compiled F+B & Compile+F+B & Speedup & Logical ratio \\"
        ),
        r"\midrule",
    ]
    for row in component["rows"]:
        timing = row["timing_benchmark"]["summary_s"]
        replay = float(timing["replay_total_forward_backward"]["median"])
        compiled = float(timing["compiled_total_forward_backward"]["median"])
        compile_total = float(
            timing["compiled_compile_plus_forward_backward"]["median"]
        )
        speedup = replay / compile_total if compile_total > 0.0 else math.inf
        cells = (
            str(int(row["frame_count"])),
            f"{float(row['image']['max_abs_error']):.3g}",
            f"{float(row['gradient']['global_normalized_l2_error']):.3g}",
            f"{float(row['atlas']['fallback_fraction']):.4f}",
            f"{replay:.4f}",
            f"{compiled:.4f}",
            f"{compile_total:.4f}",
            f"{speedup:.3f}x",
            f"{float(row['payload_bytes']['compiled_to_replay_logical_volume_ratio']):.5f}",
        )
        markdown.append("| " + " | ".join(cells) + " |")
        latex.append(" & ".join(cells) + r" \\")
    markdown.extend(
        (
            "",
            "Timing values are warmed repeated medians. “Logical-volume ratio” "
            "is the report’s tensor-element accounting and is not a storage "
            "or peak-memory claim.",
        )
    )
    latex.extend((r"\bottomrule", r"\end{tabular}"))
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    latex_path.write_text("\n".join(latex) + "\n", encoding="utf-8")


def _write_variable_table(
    component: Mapping[str, Any],
    markdown_path: Path,
    latex_path: Path,
) -> None:
    if component["accepted"] is not True:
        reason = "The verified variable-camera closure/death curve is missing."
        markdown_path.write_text(
            _placeholder_markdown("Variable-camera closure/death curve", reason),
            encoding="utf-8",
        )
        latex_path.write_text(
            _placeholder_latex("Variable-camera closure/death curve", reason),
            encoding="utf-8",
        )
        return
    markdown = [
        "# Variable-camera closure/death curve",
        "",
        "| Half span (deg) | Regime | Charts | Events (support/visibility) "
        "| Fallback samples | Invalid samples | Image max error | VJP rel. L2 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    latex = [
        r"\begin{tabular}{rlrrrrrr}",
        r"\toprule",
        (
            r"Half span & Regime & Charts & Events (S/V) & Fallback "
            r"& Invalid & Image err. & VJP rel. \\"
        ),
        r"\midrule",
    ]
    for row in component["rows"]:
        if row.get("compiled_quality_metrics_status") == (
            "structurally_unavailable_compiler_unresolved"
        ):
            reasons = ",".join(str(value) for value in row["unresolved_chart_reasons"])
            cells = (
                f"{float(row['motion_half_span_degrees']):.3g}",
                f"death ({int(row['unresolved_chart_count'])} unresolved: {reasons})",
                str(int(row["chart_count"])),
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
            )
        else:
            cells = (
                f"{float(row['motion_half_span_degrees']):.3g}",
                str(row["regime"]),
                str(int(row["chart_count"])),
                (
                    f"{int(row['support_event_count'])}/"
                    f"{int(row['visibility_event_count'])}"
                ),
                f"{float(row['fallback_sample_fraction']):.4f}",
                f"{float(row['invalid_sample_fraction']):.4f}",
                f"{float(row['image_max_abs_error']):.3g}",
                f"{float(row['world_vjp_rel_l2_max']):.3g}",
            )
        markdown.append("| " + " | ".join(cells) + " |")
        latex.append(
            " & ".join(_latex_escape(cell) for cell in cells) + r" \\"
        )
    latex.extend((r"\bottomrule", r"\end{tabular}"))
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    latex_path.write_text("\n".join(latex) + "\n", encoding="utf-8")


def _svg_axes(
    *,
    width: int,
    height: int,
    title: str,
    y_label: str,
    x_label: str,
) -> tuple[list[str], tuple[float, float, float, float]]:
    left, right, top, bottom = 90.0, 30.0, 58.0, 72.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        f'<rect width="{width}" height="{height}" fill="white"/>',
        (
            f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" '
            f'font-size="18" font-weight="bold">{html.escape(title)}</text>'
        ),
        (
            f'<line x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}" '
            'stroke="#111827"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + plot_height}" stroke="#111827"/>'
        ),
        (
            f'<text x="{width / 2:.1f}" y="{height - 16}" '
            f'text-anchor="middle" font-size="13">{html.escape(x_label)}</text>'
        ),
        (
            f'<text x="20" y="{height / 2:.1f}" '
            f'transform="rotate(-90 20 {height / 2:.1f})" '
            f'text-anchor="middle" font-size="13">{html.escape(y_label)}</text>'
        ),
    ]
    return elements, (left, top, plot_width, plot_height)


def _write_public_quality_svg(
    component: Mapping[str, Any],
    path: Path,
) -> None:
    if component["accepted"] is not True:
        path.write_text(
            _placeholder_svg(
                "Public held-out quality",
                "The expected schema-v2 matrix is incomplete.",
            ),
            encoding="utf-8",
        )
        return
    rows = component["aggregated"]
    width = max(900, 125 * len(rows) + 140)
    height = 480
    elements, (left, top, plot_width, plot_height) = _svg_axes(
        width=width,
        height=height,
        title="Held-out quality under the declared public protocols",
        y_label="PSNR (dB)",
        x_label="Protocol role / representation",
    )
    upper = max(
        1.0,
        max(
            float(row["heldout_eval_psnr_mean"])
            + float(row["heldout_eval_psnr_std"])
            for row in rows
        )
        * 1.12,
    )
    slot = plot_width / len(rows)
    colors = {
        "world_tubes": "#2563eb",
        "worldfoam": "#059669",
        "dynamic_3dgs": "#d97706",
    }
    for index, row in enumerate(rows):
        value = float(row["heldout_eval_psnr_mean"])
        std = float(row["heldout_eval_psnr_std"])
        x = left + slot * index + slot * 0.18
        bar_width = slot * 0.64
        bar_height = plot_height * value / upper
        y = top + plot_height - bar_height
        center = x + bar_width / 2.0
        error = plot_height * std / upper
        elements.extend(
            (
                (
                    f'<rect class="data-bar" x="{x:.2f}" y="{y:.2f}" '
                    f'width="{bar_width:.2f}" height="{bar_height:.2f}" '
                    f'fill="{colors[str(row["lane"])]}"/>'
                ),
                (
                    f'<line x1="{center:.2f}" y1="{y - error:.2f}" '
                    f'x2="{center:.2f}" y2="{y + error:.2f}" stroke="#111827"/>'
                ),
                (
                    f'<text x="{center:.2f}" y="{y - error - 8:.2f}" '
                    f'text-anchor="middle" font-size="11">{value:.3f}</text>'
                ),
                (
                    f'<text x="{center:.2f}" y="{top + plot_height + 20:.2f}" '
                    f'text-anchor="middle" font-size="10">'
                    f'{html.escape(str(row["role"]))}</text>'
                ),
                (
                    f'<text x="{center:.2f}" y="{top + plot_height + 35:.2f}" '
                    f'text-anchor="middle" font-size="10">'
                    f'{html.escape(str(row["lane"]))}</text>'
                ),
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _write_public_cost_svg(
    component: Mapping[str, Any],
    path: Path,
) -> None:
    if component["accepted"] is not True:
        path.write_text(
            _placeholder_svg(
                "Public cost and stored state",
                "The expected schema-v2 matrix is incomplete.",
            ),
            encoding="utf-8",
        )
        return
    primary = [
        row
        for row in component["aggregated"]
        if row["role"] == "primary_progressive"
    ]
    rows = primary or component["aggregated"]
    width, height = 960, 460
    elements, (left, top, plot_width, plot_height) = _svg_axes(
        width=width,
        height=height,
        title="Primary protocol cost and serialized state",
        y_label="Relative to maximum within each metric",
        x_label="Representation",
    )
    metrics = (
        ("train_wall_s_mean", "wall", "#7c3aed"),
        ("sampled_peak_driver_allocated_bytes_mean", "driver", "#db2777"),
        ("serialized_checkpoint_bytes_mean", "checkpoint", "#0891b2"),
    )
    maxima = {
        key: max(float(row[key]) for row in rows)
        for key, _label, _color in metrics
    }
    group_slot = plot_width / len(rows)
    bar_slot = group_slot * 0.72 / len(metrics)
    for row_index, row in enumerate(rows):
        group_left = left + row_index * group_slot + group_slot * 0.14
        for metric_index, (key, label, color) in enumerate(metrics):
            value = float(row[key])
            normalized = value / maxima[key] if maxima[key] > 0.0 else 0.0
            x = group_left + metric_index * bar_slot
            bar_height = normalized * plot_height
            y = top + plot_height - bar_height
            elements.append(
                f'<rect class="data-bar" x="{x:.2f}" y="{y:.2f}" '
                f'width="{bar_slot * 0.82:.2f}" height="{bar_height:.2f}" '
                f'fill="{color}"/>'
            )
            elements.append(
                f'<text x="{x + bar_slot * 0.41:.2f}" y="{y - 6:.2f}" '
                f'text-anchor="middle" font-size="10">{normalized:.2f}</text>'
            )
            if row_index == 0:
                elements.append(
                    f'<text x="{x + bar_slot * 0.41:.2f}" y="{top - 10:.2f}" '
                    f'text-anchor="middle" font-size="10" fill="{color}">'
                    f"{label}</text>"
                )
        center = group_left + group_slot * 0.36
        elements.append(
            f'<text x="{center:.2f}" y="{top + plot_height + 26:.2f}" '
            f'text-anchor="middle" font-size="12">'
            f'{html.escape(str(row["lane"]))}</text>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _write_frozen_svg(component: Mapping[str, Any], path: Path) -> None:
    if component["accepted"] is not True:
        path.write_text(
            _placeholder_svg(
                "Frozen-world replay versus compiled atlas",
                "The publication-eligible frozen sweep is missing.",
            ),
            encoding="utf-8",
        )
        return
    rows = component["rows"]
    width, height = 900, 460
    elements, (left, top, plot_width, plot_height) = _svg_axes(
        width=width,
        height=height,
        title="Frozen identical-world replay versus compiled atlas",
        y_label="Median forward+backward wall time (s)",
        x_label="Full-interval evaluation frames (F)",
    )
    series = (
        ("replay_total_forward_backward", "Replay", "#dc2626"),
        (
            "compiled_compile_plus_forward_backward",
            "Compiled incl. compile",
            "#2563eb",
        ),
    )
    values = [
        float(row["timing_benchmark"]["summary_s"][key]["median"])
        for row in rows
        for key, _label, _color in series
    ]
    upper = max(values) * 1.12 if values else 1.0
    min_frame = min(int(row["frame_count"]) for row in rows)
    max_frame = max(int(row["frame_count"]) for row in rows)
    log_min = math.log2(min_frame)
    log_span = max(1.0, math.log2(max_frame) - log_min)
    for key, label, color in series:
        points: list[str] = []
        for row in rows:
            frame_count = int(row["frame_count"])
            value = float(row["timing_benchmark"]["summary_s"][key]["median"])
            x = left + (math.log2(frame_count) - log_min) / log_span * plot_width
            y = top + plot_height - value / upper * plot_height
            points.append(f"{x:.2f},{y:.2f}")
            elements.extend(
                (
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>',
                    f'<text x="{x:.2f}" y="{top + plot_height + 20:.2f}" '
                    f'text-anchor="middle" font-size="10">{frame_count}</text>',
                )
            )
        elements.append(
            f'<polyline class="data-line" points="{" ".join(points)}" '
            f'fill="none" stroke="{color}" stroke-width="3"/>'
        )
        legend_x = left + 15 + (180 if label.startswith("Compiled") else 0)
        elements.extend(
            (
                f'<line x1="{legend_x}" y1="{top - 18}" '
                f'x2="{legend_x + 28}" y2="{top - 18}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 35}" y="{top - 13}" font-size="11">'
                f"{html.escape(label)}</text>",
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _write_variable_svg(component: Mapping[str, Any], path: Path) -> None:
    if component["accepted"] is not True:
        path.write_text(
            _placeholder_svg(
                "Variable-camera closure/death curve",
                "The verified bounded camera-program stress report is missing.",
            ),
            encoding="utf-8",
        )
        return
    rows = sorted(
        component["rows"],
        key=lambda row: float(row["motion_half_span_degrees"]),
    )
    width, height = 900, 460
    elements, (left, top, plot_width, plot_height) = _svg_axes(
        width=width,
        height=height,
        title="Bounded variable-camera atlas closure and death",
        y_label="Fraction / normalized error",
        x_label="Camera motion half-span (degrees)",
    )
    min_x = float(rows[0]["motion_half_span_degrees"])
    max_x = float(rows[-1]["motion_half_span_degrees"])
    x_span = max(1.0e-12, max_x - min_x)
    quality_rows = [
        row
        for row in rows
        if row.get("compiled_quality_metrics_status") == "available"
    ]
    max_error = max(float(row["image_max_abs_error"]) for row in quality_rows)
    series = (
        ("fallback_sample_fraction", "Fallback fraction", "#2563eb", 1.0),
        (
            "image_max_abs_error",
            "Image max error (normalized)",
            "#dc2626",
            max(max_error, 1.0e-12),
        ),
    )
    for key, label, color, normalizer in series:
        points = []
        for row in quality_rows:
            value = min(1.0, max(0.0, float(row[key]) / normalizer))
            x = left + (
                float(row["motion_half_span_degrees"]) - min_x
            ) / x_span * plot_width
            y = top + plot_height - value * plot_height
            points.append(f"{x:.2f},{y:.2f}")
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
            )
        elements.append(
            f'<polyline class="data-line" points="{" ".join(points)}" '
            f'fill="none" stroke="{color}" stroke-width="3"/>'
        )
        legend_x = left + 15 + (215 if key == "image_max_abs_error" else 0)
        elements.extend(
            (
                f'<line x1="{legend_x}" y1="{top - 18}" '
                f'x2="{legend_x + 28}" y2="{top - 18}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 35}" y="{top - 13}" font-size="11">'
                f"{html.escape(label)}</text>",
            )
        )
    first_death = component["summary"].get("first_death_half_span_degrees")
    if _finite_number(first_death):
        death_x = left + (float(first_death) - min_x) / x_span * plot_width
        elements.extend(
            (
                f'<line x1="{death_x:.2f}" y1="{top}" x2="{death_x:.2f}" '
                f'y2="{top + plot_height}" stroke="#7f1d1d" '
                'stroke-width="2" stroke-dasharray="6 5"/>',
                f'<text x="{death_x + 5:.2f}" y="{top + 16}" '
                'font-size="11" fill="#7f1d1d">compiler death boundary</text>',
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _write_ledger_markdown(bundle: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# World Tubes submission evidence ledger",
        "",
        f"Overall evidence-bundle status: **{bundle['status']}**.",
        "",
        (
            "This ledger covers generated evidence artifacts only. "
            "Venue conversion and the manuscript-package gate remain required."
        ),
        "",
        "| Component | Status | Accepted | Input |",
        "|---|---|---:|---|",
    ]
    for name, component in bundle["components"].items():
        input_path = component.get(
            "input",
            component.get("matrix_path", ""),
        )
        lines.append(
            f"| {name} | {component['status']} | "
            f"{'yes' if component['accepted'] else 'no'} | `{input_path}` |"
        )
    public = bundle["components"]["public_context"]
    lines.extend(
        (
            "",
            "## Public matrix slots",
            "",
            "| # | Role | Protocol | Seed | Policy | Status |",
            "|---:|---|---|---:|---|---|",
        )
    )
    for slot in public["slots"]:
        lines.append(
            f"| {slot['ordinal']} | {slot['role']} | {slot['protocol_name']} "
            f"| {slot['seed']} | {slot['world_tubes_backward_policy']} "
            f"| {slot['status']} |"
        )
        for error in slot["errors"]:
            lines.append(f"|  |  | validation error: {error} |  |  |  |")
    if bundle["missing_runtime_inputs"]:
        lines.extend(("", "## Missing runtime inputs", ""))
        for item in bundle["missing_runtime_inputs"]:
            label = item.get("run_key", item["component"])
            lines.append(
                f"- `{label}`: {item['status']} — "
                f"`{item['expected_summary']}`"
            )
    else:
        lines.extend(("", "All declared runtime inputs are accepted.", ""))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "matrix_ordinal",
        "run_key",
        "role",
        "protocol",
        "scene_sample",
        "train_cameras",
        "heldout_cameras",
        "seed",
        "lane",
        "backward_policy",
        "repository_commit",
        "star_uvt_commit",
        "dataset_input_sha256",
        "decoded_dataset_sha256",
        "evaluator_sha256",
        "runtime_sha256",
        "sample_schedule_sha256",
        "route_native_sha256",
        "wandb_run_id",
        "run_summary",
        *QUALITY_KEYS,
        *COST_KEYS,
        *TIMING_KEYS,
        "diagnostics_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


ARTIFACT_FILENAMES = (
    "evidence_ledger.json",
    "evidence_ledger.md",
    "missing_runtime_inputs.json",
    "public_context_rows.json",
    "public_context_rows.csv",
    "public_context_table.md",
    "public_context_table.tex",
    "public_heldout_quality.svg",
    "public_cost_and_storage.svg",
    "theorem_table.md",
    "theorem_table.tex",
    "frozen_scaling_table.md",
    "frozen_scaling_table.tex",
    "frozen_scaling.svg",
    "variable_camera_table.md",
    "variable_camera_table.tex",
    "variable_camera_closure_death.svg",
)


def write_bundle(bundle: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    public = bundle["components"]["public_context"]
    theorem = bundle["components"]["theorem_correctness"]
    frozen = bundle["components"]["frozen_world_scaling"]
    variable = bundle["components"]["variable_camera_closure_death"]

    (out_dir / "evidence_ledger.json").write_bytes(_json_bytes(bundle))
    _write_ledger_markdown(bundle, out_dir / "evidence_ledger.md")
    (out_dir / "missing_runtime_inputs.json").write_bytes(
        _json_bytes(
            {
                "schema_version": 1,
                "submission_ready": bundle["submission_ready"],
                "readiness_scope": bundle["readiness_scope"],
                "manuscript_package_required": bundle[
                    "manuscript_package_required"
                ],
                "inputs": bundle["missing_runtime_inputs"],
            }
        )
    )
    (out_dir / "public_context_rows.json").write_bytes(
        _json_bytes(
            {
                "schema_version": 1,
                "status": public["status"],
                "matrix": public["matrix"],
                "rows": public["rows"],
                "aggregated": public["aggregated"],
            }
        )
    )
    _write_csv(out_dir / "public_context_rows.csv", public["rows"])
    _write_public_table(
        public,
        out_dir / "public_context_table.md",
        out_dir / "public_context_table.tex",
    )
    _write_public_quality_svg(public, out_dir / "public_heldout_quality.svg")
    _write_public_cost_svg(public, out_dir / "public_cost_and_storage.svg")
    _write_theorem_table(
        theorem,
        out_dir / "theorem_table.md",
        out_dir / "theorem_table.tex",
    )
    _write_frozen_table(
        frozen,
        out_dir / "frozen_scaling_table.md",
        out_dir / "frozen_scaling_table.tex",
    )
    _write_frozen_svg(frozen, out_dir / "frozen_scaling.svg")
    _write_variable_table(
        variable,
        out_dir / "variable_camera_table.md",
        out_dir / "variable_camera_table.tex",
    )
    _write_variable_svg(
        variable,
        out_dir / "variable_camera_closure_death.svg",
    )

    artifacts = []
    for name in ARTIFACT_FILENAMES:
        artifact = out_dir / name
        artifacts.append(
            {
                "path": name,
                "bytes": artifact.stat().st_size,
                "sha256": _file_sha256(artifact),
            }
        )
    manifest = {
        "schema_version": 1,
        "generator": GENERATOR_NAME,
        "status": bundle["status"],
        "submission_ready": bundle["submission_ready"],
        "readiness_scope": bundle["readiness_scope"],
        "manuscript_package_required": bundle[
            "manuscript_package_required"
        ],
        "ledger_sha256": bundle["ledger_sha256"],
        "artifacts": artifacts,
    }
    manifest["manifest_payload_sha256"] = _canonical_json_sha256(manifest)
    (out_dir / "artifact_manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def verify_bundle_dir(
    out_dir: Path,
    *,
    require_complete: bool = True,
) -> list[str]:
    errors: list[str] = []
    manifest_path = out_dir / "artifact_manifest.json"
    ledger_path = out_dir / "evidence_ledger.json"
    if not manifest_path.is_file():
        return ["artifact_manifest.json is missing"]
    if not ledger_path.is_file():
        return ["evidence_ledger.json is missing"]
    try:
        manifest = _load_json(manifest_path)
        ledger = _load_json(ledger_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [str(error)]
    manifest_digest = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if manifest_digest != _canonical_json_sha256(payload):
        errors.append("artifact manifest payload digest is invalid")
    ledger_digest = ledger.get("ledger_sha256")
    ledger_payload = dict(ledger)
    ledger_payload.pop("ledger_sha256", None)
    if ledger_digest != _canonical_json_sha256(ledger_payload):
        errors.append("evidence ledger digest is invalid")
    if manifest.get("ledger_sha256") != ledger_digest:
        errors.append("manifest and ledger digests disagree")
    for payload_name, payload in (("manifest", manifest), ("ledger", ledger)):
        if payload.get("readiness_scope") != "evidence_artifact_bundle_only":
            errors.append(f"{payload_name} readiness scope is ambiguous")
        if payload.get("manuscript_package_required") is not True:
            errors.append(
                f"{payload_name} must require the separate manuscript package gate"
            )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("artifact list is invalid")
        artifacts = []
    expected_names = set(ARTIFACT_FILENAMES)
    actual_names = {
        artifact.get("path")
        for artifact in artifacts
        if isinstance(artifact, Mapping)
    }
    if actual_names != expected_names:
        errors.append("artifact file set is incomplete or unexpected")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        name = artifact.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            errors.append("artifact path is not a local filename")
            continue
        path = out_dir / name
        if not path.is_file():
            errors.append(f"{name} is missing")
            continue
        if int(artifact.get("bytes", -1)) != path.stat().st_size:
            errors.append(f"{name} byte count drifted")
        if artifact.get("sha256") != _file_sha256(path):
            errors.append(f"{name} digest drifted")
    if require_complete and (
        ledger.get("submission_ready") is not True
        or manifest.get("submission_ready") is not True
        or ledger.get("status") != "complete"
        or manifest.get("status") != "complete"
    ):
        errors.append("artifact evidence bundle is not complete")
    return sorted(set(errors))


def verify_manuscript_package(
    *,
    bundle_dir: Path,
    draft_path: Path = DEFAULT_PAPER_DRAFT,
    tex_path: Path = DEFAULT_PAPER_TEX,
    bibliography_path: Path = DEFAULT_PAPER_BIBLIOGRAPHY,
    require_complete: bool = True,
) -> list[str]:
    errors = verify_bundle_dir(
        bundle_dir,
        require_complete=require_complete,
    )
    sources: dict[str, str] = {}
    for label, path in (
        ("paper draft", draft_path),
        ("generated TeX", tex_path),
        ("bibliography", bibliography_path),
    ):
        if not path.is_file():
            errors.append(f"{label} is missing: {_display_path(path)}")
            continue
        sources[label] = path.read_text(encoding="utf-8")
    draft = sources.get("paper draft", "")
    tex = sources.get("generated TeX", "")
    bibliography = sources.get("bibliography", "")
    for artifact_path in MANUSCRIPT_TABLE_INPUTS:
        directive = rf"\input{{{artifact_path}}}"
        if directive not in draft:
            errors.append(f"paper draft does not input {artifact_path}")
        if directive not in tex:
            errors.append(f"generated TeX does not input {artifact_path}")
        if not (ROOT / artifact_path).is_file():
            errors.append(f"manuscript table fragment is missing: {artifact_path}")
    for token in FORBIDDEN_MANUSCRIPT_EVIDENCE:
        if token in draft:
            errors.append(f"paper draft contains forbidden stale evidence: {token}")
        if token in tex:
            errors.append(f"generated TeX contains forbidden stale evidence: {token}")

    citation_keys = set(re.findall(r"@([A-Za-z0-9][A-Za-z0-9_.:+-]*)", draft))
    bibliography_keys = set(
        re.findall(
            r"@[A-Za-z]+\s*\{\s*([^,\s]+)",
            bibliography,
            flags=re.IGNORECASE,
        )
    )
    for key in sorted(citation_keys - bibliography_keys):
        errors.append(f"bibliography is missing cited key: {key}")

    image_targets = re.findall(
        r"!\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)",
        draft,
        flags=re.DOTALL,
    )
    for target in image_targets:
        if re.match(r"^(?:https?://|data:|#)", target):
            continue
        candidate = Path(target)
        candidates = (
            (candidate,)
            if candidate.is_absolute()
            else (ROOT / candidate, draft_path.parent / candidate)
        )
        if not any(path.is_file() for path in candidates):
            errors.append(f"paper image is missing: {target}")

    try:
        ledger = _load_json(bundle_dir / "evidence_ledger.json")
    except (OSError, ValueError, json.JSONDecodeError):
        ledger = {}
    theorem_rows = (
        ledger.get("components", {})
        .get("theorem_correctness", {})
        .get("rows", [])
    )
    for row in theorem_rows if isinstance(theorem_rows, list) else ():
        metric = str(row.get("metric", "")).lower()
        if "forward ratio" in metric or "backward ratio" in metric or "timing" in metric:
            errors.append(
                "theorem table contains a timing metric reserved for the "
                "frozen-world component"
            )
    if require_complete and re.search(
        r"\\documentclass\s*\[\s*\]\s*\{article\}",
        tex,
        flags=re.DOTALL,
    ):
        errors.append(
            "generated TeX is still the generic Pandoc article, not a venue package"
        )
    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--run-root",
        type=Path,
        help=(
            "Canonical matrix output root. Defaults to output_root from the "
            "selected matrix config."
        ),
    )
    parser.add_argument(
        "--matrix-summary",
        type=Path,
        help=(
            "Canonical completed matrix_summary.json. Defaults to "
            "<run-root>/matrix_summary.json."
        ),
    )
    parser.add_argument(
        "--theorem-summary",
        type=Path,
        default=DEFAULT_THEOREM_SUMMARY,
    )
    parser.add_argument(
        "--frozen-summary",
        type=Path,
        default=DEFAULT_FROZEN_SUMMARY,
    )
    parser.add_argument(
        "--variable-camera-summary",
        type=Path,
        default=DEFAULT_VARIABLE_CAMERA_SUMMARY,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write and verify honest placeholders instead of exiting after generation.",
    )
    parser.add_argument("--verify-dir", type=Path)
    parser.add_argument(
        "--verify-manuscript",
        action="store_true",
        help=(
            "Verify citations, local media, generated table inputs, stale "
            "evidence exclusion, and the artifact bundle without importing "
            "Torch."
        ),
    )
    args = parser.parse_args()

    if args.verify_manuscript:
        bundle_dir = (
            args.verify_dir.resolve()
            if args.verify_dir is not None
            else args.out_dir.resolve()
        )
        errors = verify_manuscript_package(
            bundle_dir=bundle_dir,
            require_complete=not args.allow_incomplete,
        )
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"Verified manuscript package at {bundle_dir}")
        return

    if args.verify_dir is not None:
        errors = verify_bundle_dir(
            args.verify_dir.resolve(),
            require_complete=not args.allow_incomplete,
        )
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"Verified {args.verify_dir.resolve()}")
        return

    matrix_path = args.matrix.resolve()
    run_root = resolve_matrix_run_root(matrix_path, args.run_root)
    matrix_summary = (
        args.matrix_summary.resolve()
        if args.matrix_summary is not None
        else (run_root / "matrix_summary.json")
    )
    bundle = build_bundle(
        matrix_path=matrix_path,
        run_root=run_root,
        matrix_summary=matrix_summary,
        theorem_summary=args.theorem_summary.resolve(),
        frozen_summary=args.frozen_summary.resolve(),
        variable_camera_summary=args.variable_camera_summary.resolve(),
    )
    out_dir = args.out_dir.resolve()
    write_bundle(bundle, out_dir)
    errors = verify_bundle_dir(
        out_dir,
        require_complete=not args.allow_incomplete,
    )
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
