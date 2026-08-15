"""Fail-closed production boundary for one WorldFoam G4 public-quality row.

The row worker owns everything that must be identical across representations:
the frozen protocol, deterministic spacetime sampler, exhaustive pixel-chunk
schedule, public train/heldout cache contract, final-checkpoint evaluator, and
artifact receipts.  Route executors own only real training and rendering.

This module deliberately imports neither Torch nor a native extension during
preflight.  Missing mapped caches, evaluator assets, runtime attestation, or a
production route executor therefore abort before accelerator allocation.  No
procedural, fake-native, source-only, smoke, or reduced-pixel path can emit a
``g4_row.json``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from config_utils import load_config_file, serialize_config_value
from paper_training_protocol import (
    PaperRGBMetricAccumulator,
    PaperSampleScheduleDigest,
    SpacetimeEpochSampler,
    lpips_alex_asset_status,
    paper_evaluator_contract,
    resolve_paper_training_protocol,
)
from paper_training_types import PaperStage, PaperTrainingProtocol, SpacetimeBatch
from worldfoam_g4_tractability import audit_worldfoam_g4_full_schedule


ROOT = Path(__file__).resolve().parents[2]
LANE2 = ROOT / "research_experiments" / "world_foam_lane2"
for _import_root in (ROOT, LANE2):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from verify_worldfoam_public_quality_ablation import (  # noqa: E402
    DEFAULT_CONFIG,
    REQUIRED_COST,
    REQUIRED_METRICS,
    REQUIRED_ROUTES,
    ROW_KEYS,
    ROW_KIND,
    canonical_sha256,
    file_sha256,
    validate_contract,
)


ROW_SCHEMA_VERSION = 1
WORK_PLAN_SCHEMA_VERSION = 1
DATASET_CAPABILITY_SCHEMA_VERSION = 1
MAXIMUM_PIXELS_PER_CHUNK = 32_768
EXPECTED_TARGET_PIXELS = 235_929_600
EXPECTED_SAMPLED_IMAGES = 1_200
EXPECTED_HELDOUT_FRAMES = 300
EXPECTED_IMAGE_SIZE = (384, 512)
RUNTIME_CAPABILITY_PATH = (
    ROOT / "src" / "train" / "worldfoam_native4d_public_quality_capabilities.json"
)


def _process_lifetime_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024
ROUTE_EXECUTOR_MODULES = {
    "worldfoam_native4d": "worldfoam_native4d_public_quality_executor",
    "worldfoam_framewise_replay": "worldfoam_native4d_public_quality_executor",
    "world_tubes": "world_tubes_public_quality_executor",
    "dynamic_3dgs": "dynamic_3dgs_public_quality_executor",
}
ROUTE_EXECUTION_MODES = {
    "worldfoam_native4d": "compiled_shared_adjoint",
    "worldfoam_framewise_replay": "framewise_same_representation",
    "world_tubes": "selected_time_uvt_replay",
    "dynamic_3dgs": "per_frame_dynamic_splats",
}
REQUIRED_RUNTIME_CAPABILITIES = {
    "schema_version": 1,
    "status": "runtime_verified",
    "row_kind": ROW_KIND,
    "supported_routes": list(REQUIRED_ROUTES),
    "real_native_only": True,
    "public_neural3d_targets": True,
    "heldout_camera_evaluation": True,
    "full_temporal_evaluation": True,
    "compiled_shared_adjoint": True,
    "same_representation_framewise_replay": True,
    "final_checkpoint_metrics": True,
    "wandb_run_file": True,
    "proxy_or_fake_native_permitted": False,
    "smoke_as_public_evidence_permitted": False,
}

_DATASET_CAPABILITY_KEYS = {
    "schema_version",
    "kind",
    "scene",
    "sample_id",
    "protocol_sha256",
    "dataset_manifest_sha256",
    "frame_count",
    "image_size",
    "train_cameras",
    "heldout_cameras",
    "train_binding",
    "heldout_binding",
    "provider_factory",
    "public_data",
    "calibrated_multiview",
    "selected_pixel_reads",
    "full_frame_materialization_required",
    "initialization_sha256",
    "compiler_sha256",
    "worldfoam_runtime_sha256",
    "capability_sha256",
}
_BOUND_FILE_KEYS = {"path", "bytes", "sha256", "target_split", "camera_ids"}
_FACTORY_KEYS = {"module", "callable", "source_path", "source_sha256"}
_INITIALIZATION_KEYS = {"source_asset", "initializer", "builder"}
_SOURCE_ASSET_KEYS = {
    "path",
    "expected_bytes",
    "expected_sha256",
    "minimum_point_count",
}
_INITIALIZER_KEYS = {
    "source_path",
    "source_coordinate_frame",
    "point_transform",
    "maximum_source_asset_bytes",
    "maximum_source_point_count",
    "site_count",
    "sample_mode",
    "sample_seed",
    "coordinate_quantization_step",
    "weight_coefficients",
    "weight_quantization_step",
    "initial_density",
}
_BUILDER_KEYS = {"script_path", "argv", "deterministic_output_required"}
_COMPILER_KEYS = {
    "near",
    "far",
    "node_count",
    "maximum_sites_per_track_compile",
    "maximum_charts_per_track",
    "maximum_owner_runs_per_chart",
    "rank_selection_provenance",
}
_WORLD_FOAM_RUNTIME_KEYS = {
    "schema_version",
    "background_rgb",
    "shared_reverse_mode",
    "maximum_samples_per_launch",
    "maximum_artifact_accounted_bytes_per_entry",
    "maximum_material_state_logical_tensor_bytes",
    "maximum_tracks_per_bundle",
    "maximum_observations_per_bundle",
    "maximum_rows_per_native_block",
    "cone_tolerance",
    "material_parameterization",
    "material_sgd_policy",
    "artifact_store_policy",
    "lazy_memory_policy",
    "full_geometry_memory_policy",
    "combined_sgd_policy",
}
_MATERIAL_PARAMETERIZATION_KEYS = {
    "density_beta",
    "density_threshold",
    "minimum_density",
    "color_epsilon",
}
_MATERIAL_SGD_POLICY_KEYS = {
    "color_learning_rate",
    "density_learning_rate",
    "maximum_absolute_raw_color_value",
    "maximum_absolute_raw_density_value",
}
_ARTIFACT_STORE_POLICY_KEYS = {
    "maximum_entries",
    "maximum_resident_accounted_bytes",
}
_LAZY_MEMORY_POLICY_KEYS = {
    "max_global_material_and_bar_tensor_bytes",
    "max_bundle_observation_count",
    "max_lane_resident_logical_tensor_bytes",
    "max_active_node_and_vjp_tensor_bytes",
    "max_decoded_frame_scratch_tensor_bytes",
    "max_selected_frame_target_tensor_bytes",
    "max_sample_launch_tensor_bytes",
    "max_coordinator_visible_live_tensor_bytes",
    "target_frame_access_mode",
    "max_step_target_frame_cache_tensor_bytes",
}
_FULL_GEOMETRY_MEMORY_POLICY_KEYS = {
    "maximum_global_geometry_bar_logical_tensor_bytes",
    "maximum_geometry_bridge_visible_peak_logical_tensor_bytes",
    "maximum_fused_union_transaction_scratch_tensor_bytes",
}
_COMBINED_SGD_POLICY_KEYS = {
    "position_learning_rate",
    "velocity_learning_rate",
    "weight_learning_rate",
    "maximum_absolute_position_update",
    "maximum_absolute_velocity_update",
    "maximum_absolute_weight_update",
    "maximum_absolute_position_value",
    "maximum_absolute_velocity_value",
    "maximum_absolute_weight_value",
    "maximum_combined_state_logical_tensor_bytes",
    "maximum_update_candidate_logical_tensor_bytes",
    "maximum_candidate_world_geometry_clone_logical_tensor_bytes",
    "maximum_update_validation_scratch_logical_tensor_bytes",
    "maximum_old_candidate_authorization_logical_tensor_bytes",
    "maximum_checkpoint_logical_tensor_bytes",
    "maximum_state_checkpoint_logical_tensor_bytes",
    "maximum_state_checkpoint_payload_logical_tensor_bytes",
    "maximum_transaction_tracked_logical_and_store_accounted_bytes",
    "maximum_recompile_request_count",
    "maximum_recompile_track_id_logical_bytes",
    "maximum_artifact_accounted_bytes",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        serialize_config_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _repo_path(value: str | Path, *, name: str, must_exist: bool = False) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{name} leaves the repository") from error
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"{name} is missing: {resolved}")
    return resolved


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected one JSON object in {path}")
    return payload


def _file_identity(path: Path, **extra: Any) -> dict[str, Any]:
    resolved = _repo_path(path, name="artifact", must_exist=True)
    if resolved.stat().st_size < 1:
        raise ValueError(f"artifact is empty: {resolved}")
    return {
        "path": _display(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
        **extra,
    }


@dataclass(frozen=True)
class PixelChunkRequest:
    split: str
    step: int | None
    sample_slot: int | None
    camera_index: int
    frame_index: int
    pixel_start: int
    pixel_count: int
    image_height: int
    image_width: int
    # ``None`` retains the frozen G4-v1 contiguous interval semantics.  G4-v2
    # uses ``pixel_start`` as the offset inside its selected-pixel sequence and
    # carries the actual ascending sensor ids here.  Keeping the optional field
    # out of ``as_dict`` for v1 preserves every v1 schedule digest byte.
    pixel_ids: tuple[int, ...] | None = None

    @property
    def pixel_stop(self) -> int:
        return self.pixel_start + self.pixel_count

    def as_dict(self) -> dict[str, Any]:
        result = {
            "split": self.split,
            "step": self.step,
            "sample_slot": self.sample_slot,
            "camera_index": self.camera_index,
            "frame_index": self.frame_index,
            "pixel_start": self.pixel_start,
            "pixel_count": self.pixel_count,
            "image_height": self.image_height,
            "image_width": self.image_width,
        }
        if self.pixel_ids is not None:
            result["pixel_ids"] = list(self.pixel_ids)
        return result


@dataclass(frozen=True)
class StepWork:
    step: int
    stage: PaperStage
    batch: SpacetimeBatch


@dataclass(frozen=True)
class FullPixelWorkPlan:
    protocol: PaperTrainingProtocol
    seed: int
    sampler_seed: int
    steps: tuple[StepWork, ...]
    spacetime_schedule_sha256: str
    pixel_chunk_manifest_sha256: str
    sample_schedule_sha256: str
    sampled_image_count: int
    pixel_chunk_count: int
    target_pixels: int
    maximum_pixels_per_chunk: int

    def iter_step_training_chunks(
        self,
        work: StepWork,
    ) -> Iterator[PixelChunkRequest]:
        height = work.stage.image_size.height
        width = work.stage.image_size.width
        pixels = work.stage.image_size.pixels
        for sample_slot, sample in enumerate(work.batch.samples):
            for pixel_start in range(0, pixels, self.maximum_pixels_per_chunk):
                yield PixelChunkRequest(
                    split="train",
                    step=work.step,
                    sample_slot=sample_slot,
                    camera_index=sample.view_index,
                    frame_index=sample.frame_index,
                    pixel_start=pixel_start,
                    pixel_count=min(
                        self.maximum_pixels_per_chunk,
                        pixels - pixel_start,
                    ),
                    image_height=height,
                    image_width=width,
                )

    def iter_training_chunks(self) -> Iterator[PixelChunkRequest]:
        for work in self.steps:
            yield from self.iter_step_training_chunks(work)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORK_PLAN_SCHEMA_VERSION,
            "algorithm": "spacetime_epoch_all_pixels_chunked_v1",
            "sampler_seed": self.sampler_seed,
            "optimizer_steps": len(self.steps),
            "sampled_image_count": self.sampled_image_count,
            "pixel_chunk_count": self.pixel_chunk_count,
            "target_pixels": self.target_pixels,
            "maximum_pixels_per_chunk": self.maximum_pixels_per_chunk,
            "spacetime_schedule_sha256": self.spacetime_schedule_sha256,
            "pixel_chunk_manifest_sha256": self.pixel_chunk_manifest_sha256,
            "sample_schedule_sha256": self.sample_schedule_sha256,
        }


def build_full_pixel_work_plan(
    protocol: PaperTrainingProtocol,
    *,
    seed: int,
    maximum_pixels_per_chunk: int = MAXIMUM_PIXELS_PER_CHUNK,
) -> FullPixelWorkPlan:
    if maximum_pixels_per_chunk < 1:
        raise ValueError("maximum_pixels_per_chunk must be positive")
    sampler_seed = int(seed) + int(protocol.sampler_seed_offset)
    sampler = SpacetimeEpochSampler(
        view_count=len(protocol.dataset.train_cameras),
        frame_indices=range(protocol.dataset.frame_count),
        batch_size=max(stage.frames_per_step for stage in protocol.stages),
        same_time_count=protocol.same_time_count,
        local_time_count=protocol.local_time_count,
        local_time_radius=protocol.local_time_radius,
        seed=sampler_seed,
    )
    schedule = PaperSampleScheduleDigest(sampler_seed=sampler_seed)
    pixel_digest = hashlib.sha256()
    steps: list[StepWork] = []
    sampled_images = 0
    pixel_chunks = 0
    target_pixels = 0
    for step in range(protocol.steps):
        stage = next(stage for stage in protocol.stages if stage.contains(step))
        batch = sampler.next_batch(stage.frames_per_step)
        schedule.record(step=step, stage=stage, batch=batch)
        work = StepWork(step=step, stage=stage, batch=batch)
        steps.append(work)
        for sample_slot, sample in enumerate(batch.samples):
            sampled_images += 1
            image_pixels = stage.image_size.pixels
            covered = 0
            for pixel_start in range(0, image_pixels, maximum_pixels_per_chunk):
                count = min(maximum_pixels_per_chunk, image_pixels - pixel_start)
                record = PixelChunkRequest(
                    split="train",
                    step=step,
                    sample_slot=sample_slot,
                    camera_index=sample.view_index,
                    frame_index=sample.frame_index,
                    pixel_start=pixel_start,
                    pixel_count=count,
                    image_height=stage.image_size.height,
                    image_width=stage.image_size.width,
                )
                pixel_digest.update(_canonical_bytes(record.as_dict()))
                pixel_digest.update(b"\n")
                covered += count
                pixel_chunks += 1
            if covered != image_pixels:
                raise ArithmeticError("pixel chunks do not exactly cover one sampled image")
            target_pixels += covered
    schedule_receipt = schedule.snapshot()
    combined_schedule = _sha256(
        {
            "schema_version": WORK_PLAN_SCHEMA_VERSION,
            "spacetime_schedule_sha256": schedule_receipt["sha256"],
            "pixel_chunk_manifest_sha256": pixel_digest.hexdigest(),
            "target_pixels": target_pixels,
            "maximum_pixels_per_chunk": maximum_pixels_per_chunk,
        }
    )
    result = FullPixelWorkPlan(
        protocol=protocol,
        seed=int(seed),
        sampler_seed=sampler_seed,
        steps=tuple(steps),
        spacetime_schedule_sha256=str(schedule_receipt["sha256"]),
        pixel_chunk_manifest_sha256=pixel_digest.hexdigest(),
        sample_schedule_sha256=combined_schedule,
        sampled_image_count=sampled_images,
        pixel_chunk_count=pixel_chunks,
        target_pixels=target_pixels,
        maximum_pixels_per_chunk=maximum_pixels_per_chunk,
    )
    if (
        protocol.steps == 300
        and protocol.final_stage.image_size.as_list() == [384, 512]
        and max(stage.frames_per_step for stage in protocol.stages) == 4
        and (
            result.target_pixels != EXPECTED_TARGET_PIXELS
            or result.sampled_image_count != EXPECTED_SAMPLED_IMAGES
        )
    ):
        raise ArithmeticError("frozen G4 all-pixel budget changed")
    return result


@dataclass(frozen=True)
class RowRequest:
    config_path: Path
    protocol_path: Path
    scene: str
    seed: int
    route: str
    output_path: Path
    allow_local_mps_execution: bool
    dataset_capability_path: Path | None = None


@dataclass(frozen=True)
class RowContext:
    request: RowRequest
    config: Mapping[str, Any]
    config_receipt: Mapping[str, Any]
    protocol: PaperTrainingProtocol
    route_spec: Mapping[str, Any]
    scene_receipt: Mapping[str, Any]
    work_plan: FullPixelWorkPlan
    source_commit: str
    dataset_capability: Mapping[str, Any]


def default_dataset_capability_path(protocol: PaperTrainingProtocol) -> Path:
    return (
        ROOT
        / "outputs"
        / "cache"
        / "worldfoam_public_quality"
        / protocol.dataset.sample_id
        / "public_train_heldout_capability.json"
    )


def _source_identity() -> dict[str, Any]:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=ROOT,
            text=True,
        ).strip()
    )
    return {"repository_commit": commit, "repository_dirty": dirty}


def _route_specs(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    routes = config.get("routes")
    if not isinstance(routes, list):
        raise ValueError("G4 routes must be a list")
    return {str(_mapping(row, name="route")["route"]): row for row in routes}


def _scene_specs(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    scenes = config.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("G4 scenes must be a list")
    return {str(_mapping(row, name="scene")["scene"]): row for row in scenes}


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _validate_initialization_contract(
    value: Any,
    *,
    primitive_count: int,
) -> dict[str, Any]:
    contract = dict(_mapping(value, name="scene initialization"))
    if set(contract) != _INITIALIZATION_KEYS:
        raise ValueError("scene initialization keys changed")
    asset = dict(_mapping(contract["source_asset"], name="source asset"))
    initializer = dict(_mapping(contract["initializer"], name="initializer"))
    builder = dict(_mapping(contract["builder"], name="initializer builder"))
    if set(asset) != _SOURCE_ASSET_KEYS:
        raise ValueError("source asset keys changed")
    if set(initializer) != _INITIALIZER_KEYS:
        raise ValueError("point-cloud initializer keys changed")
    if set(builder) != _BUILDER_KEYS:
        raise ValueError("initializer builder keys changed")
    asset_path = _repo_path(str(asset.get("path", "")), name="source asset")
    initializer_path = _repo_path(
        str(initializer.get("source_path", "")),
        name="initializer source path",
    )
    if asset_path != initializer_path:
        raise ValueError("initializer source path differs from the bound source asset")
    expected_bytes = asset.get("expected_bytes")
    if expected_bytes is not None:
        _positive_int(expected_bytes, name="source asset expected_bytes")
    expected_sha256 = asset.get("expected_sha256")
    if expected_sha256 is not None and not _valid_sha256(expected_sha256):
        raise ValueError("source asset expected_sha256 is invalid")
    minimum_points = _positive_int(
        asset.get("minimum_point_count"), name="source asset minimum_point_count"
    )
    site_count = _positive_int(initializer.get("site_count"), name="initializer site_count")
    if minimum_points != primitive_count or site_count != primitive_count:
        raise ValueError("source asset and initializer must preserve the frozen primitive count")
    if initializer.get("source_coordinate_frame") != "model":
        raise ValueError("G4 point cloud must already use the model coordinate frame")
    if initializer.get("point_transform") is not None:
        raise ValueError("model-frame G4 point clouds must not add an implicit transform")
    if initializer.get("sample_mode") != "sha256_rank":
        raise ValueError("G4 initializer must use deterministic content-ranked sampling")
    if isinstance(initializer.get("sample_seed"), bool) or not isinstance(
        initializer.get("sample_seed"), int
    ):
        raise ValueError("G4 initializer sample_seed must be an integer")
    for key in ("maximum_source_asset_bytes", "maximum_source_point_count"):
        _positive_int(initializer.get(key), name=f"initializer {key}")
    if initializer["maximum_source_asset_bytes"] < (expected_bytes or 1):
        raise ValueError("initializer byte bound is below the declared asset")
    if initializer["maximum_source_point_count"] < primitive_count:
        raise ValueError("initializer point bound is below the frozen primitive count")
    _finite_float(
        initializer.get("coordinate_quantization_step"),
        name="initializer coordinate_quantization_step",
    )
    weights = initializer.get("weight_coefficients")
    if not isinstance(weights, list) or not 1 <= len(weights) <= 3:
        raise ValueError("initializer weight_coefficients must contain one to three scalars")
    for index, weight in enumerate(weights):
        _finite_float(weight, name=f"initializer weight_coefficients[{index}]")
    _finite_float(
        initializer.get("weight_quantization_step"),
        name="initializer weight_quantization_step",
    )
    if _finite_float(
        initializer.get("initial_density"), name="initializer initial_density"
    ) <= 0.0:
        raise ValueError("initializer initial_density must be positive")
    script_path = _repo_path(
        str(builder.get("script_path", "")),
        name="initializer builder script",
        must_exist=True,
    )
    argv = builder.get("argv")
    if (
        script_path.name
        not in {
            "build_multiview_feature_triangulation_point_cloud.py",
            "prepare_ex4dgs_anchor_point_cloud.py",
        }
        or not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
        or builder.get("deterministic_output_required") is not True
    ):
        raise ValueError("initializer builder recipe is incomplete or changed")
    return {
        "source_asset": asset,
        "initializer": initializer,
        "builder": builder,
    }


def _validate_compiler_contract(value: Any, *, primitive_count: int) -> dict[str, Any]:
    compiler = dict(_mapping(value, name="G4 compiler"))
    if set(compiler) != _COMPILER_KEYS:
        raise ValueError("G4 compiler keys changed")
    near = _finite_float(compiler.get("near"), name="compiler near")
    far = _finite_float(compiler.get("far"), name="compiler far")
    if near < 0.0 or far <= near:
        raise ValueError("G4 compiler requires 0 <= near < far")
    for key in (
        "node_count",
        "maximum_sites_per_track_compile",
        "maximum_charts_per_track",
        "maximum_owner_runs_per_chart",
    ):
        _positive_int(compiler.get(key), name=f"compiler {key}")
    if compiler["node_count"] < 2:
        raise ValueError("G4 compiler node_count must be at least two")
    if compiler["maximum_sites_per_track_compile"] != primitive_count:
        raise ValueError("G4 compiler site bound changed from the frozen primitive count")
    provenance = compiler.get("rank_selection_provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("G4 compiler rank-selection provenance is missing")
    return compiler


def _validate_worldfoam_runtime_contract(value: Any) -> dict[str, Any]:
    runtime = dict(_mapping(value, name="WorldFoam G4 runtime"))
    if set(runtime) != _WORLD_FOAM_RUNTIME_KEYS or runtime.get("schema_version") != 1:
        raise ValueError("WorldFoam G4 runtime keys or schema changed")
    nested_keys = {
        "material_parameterization": _MATERIAL_PARAMETERIZATION_KEYS,
        "material_sgd_policy": _MATERIAL_SGD_POLICY_KEYS,
        "artifact_store_policy": _ARTIFACT_STORE_POLICY_KEYS,
        "lazy_memory_policy": _LAZY_MEMORY_POLICY_KEYS,
        "full_geometry_memory_policy": _FULL_GEOMETRY_MEMORY_POLICY_KEYS,
        "combined_sgd_policy": _COMBINED_SGD_POLICY_KEYS,
    }
    for name, expected in nested_keys.items():
        nested = _mapping(runtime.get(name), name=name)
        if set(nested) != expected:
            raise ValueError(f"WorldFoam G4 {name} keys changed")
    background = runtime.get("background_rgb")
    if not isinstance(background, list) or len(background) != 3:
        raise ValueError("WorldFoam G4 background must contain three scalars")
    for index, channel in enumerate(background):
        if not 0.0 <= _finite_float(channel, name=f"background[{index}]") <= 1.0:
            raise ValueError("WorldFoam G4 background left [0,1]")
    if runtime.get("shared_reverse_mode") != "fused_union_v2":
        raise ValueError("WorldFoam G4 shared reverse mode changed")
    for key in (
        "maximum_samples_per_launch",
        "maximum_artifact_accounted_bytes_per_entry",
        "maximum_material_state_logical_tensor_bytes",
        "maximum_tracks_per_bundle",
        "maximum_observations_per_bundle",
        "maximum_rows_per_native_block",
    ):
        _positive_int(runtime.get(key), name=f"WorldFoam runtime {key}")
    cone = _finite_float(runtime.get("cone_tolerance"), name="cone_tolerance")
    if cone <= 0.0:
        raise ValueError("WorldFoam G4 cone tolerance must be positive")
    lazy = runtime["lazy_memory_policy"]
    for key, item in lazy.items():
        if key == "target_frame_access_mode":
            if item != "stream_once_no_step_cache":
                raise ValueError("WorldFoam targets must use one-pass streaming")
        elif key == "max_step_target_frame_cache_tensor_bytes":
            if item != 0:
                raise ValueError("WorldFoam one-pass targets require zero frame cache")
        else:
            _positive_int(item, name=f"lazy memory {key}")
    for name in (
        "artifact_store_policy",
        "full_geometry_memory_policy",
        "combined_sgd_policy",
    ):
        for key, item in runtime[name].items():
            if "learning_rate" in key or "absolute_" in key:
                if _finite_float(item, name=f"{name}.{key}") <= 0.0:
                    raise ValueError(f"{name}.{key} must be positive")
            else:
                _positive_int(item, name=f"{name}.{key}")
    for name in ("material_parameterization", "material_sgd_policy"):
        for key, item in runtime[name].items():
            number = _finite_float(item, name=f"{name}.{key}")
            if name == "material_sgd_policy" and number <= 0.0:
                raise ValueError(f"{name}.{key} must be positive")
    return runtime


def _ascii_ply_declared_point_count(path: Path) -> int:
    header_lines: list[bytes] = []
    header_bytes = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline(64 * 1024 + 1)
            if not line:
                raise ValueError("point-cloud PLY header has no end_header marker")
            header_bytes += len(line)
            if header_bytes > 64 * 1024:
                raise ValueError("point-cloud PLY header exceeds 64 KiB")
            header_lines.append(line)
            if line == b"end_header\n":
                break
    raw = b"".join(header_lines)
    try:
        header = raw.decode("ascii", errors="strict").split("end_header\n", 1)[0]
    except UnicodeDecodeError as error:
        raise ValueError("G4 point-cloud asset must be an ASCII PLY") from error
    if not header.startswith("ply\nformat ascii 1.0\n"):
        raise ValueError("G4 point-cloud asset must use ASCII PLY 1.0")
    matches = re.findall(r"^element vertex ([0-9]+)$", header, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("G4 point-cloud asset has no unique vertex declaration")
    return int(matches[0])


def _initialization_blockers(
    initialization: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    asset = initialization["source_asset"]
    path = _repo_path(str(asset["path"]), name="source asset")
    details: dict[str, Any] = {"path": _display(path)}
    blockers: list[str] = []
    if not path.is_file():
        blockers.append(f"point_cloud_asset_missing:{_display(path)}")
        return blockers, details
    details.update(
        {
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    )
    if asset["expected_bytes"] is None or asset["expected_sha256"] is None:
        blockers.append("point_cloud_asset_identity_unsealed")
    else:
        if int(asset["expected_bytes"]) != path.stat().st_size:
            blockers.append("point_cloud_asset_bytes_drifted")
        if str(asset["expected_sha256"]) != details["sha256"]:
            blockers.append("point_cloud_asset_sha256_drifted")
    try:
        point_count = _ascii_ply_declared_point_count(path)
    except Exception as error:
        blockers.append(f"point_cloud_asset_invalid:{type(error).__name__}:{error}")
    else:
        details["declared_point_count"] = point_count
        if point_count < int(asset["minimum_point_count"]):
            blockers.append(
                "point_cloud_asset_has_insufficient_points:"
                f"{point_count}<{asset['minimum_point_count']}"
            )
    return blockers, details


def _expected_output_path(config: Mapping[str, Any], request: RowRequest) -> Path:
    return (
        _repo_path(str(config["output_root"]), name="G4 output root")
        / request.scene
        / f"seed_{request.seed}"
        / request.route
        / "g4_row.json"
    )


def resolve_row_request(request: RowRequest) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    PaperTrainingProtocol,
    Mapping[str, Any],
    Mapping[str, Any],
    FullPixelWorkPlan,
]:
    config_path = _repo_path(request.config_path, name="G4 config", must_exist=True)
    protocol_path = _repo_path(request.protocol_path, name="G4 protocol", must_exist=True)
    config = load_config_file(config_path)
    config_receipt = validate_contract(config, config_path=config_path)
    if request.scene not in config_receipt["scenes"]:
        raise ValueError("requested scene is outside the frozen G4 matrix")
    if request.seed not in config["seeds"]:
        raise ValueError("requested seed is outside the frozen G4 matrix")
    if request.route not in REQUIRED_ROUTES:
        raise ValueError("requested route is outside the frozen G4 matrix")
    route_spec = _route_specs(config)[request.route]
    if route_spec.get("execution_mode") != ROUTE_EXECUTION_MODES[request.route]:
        raise ValueError("requested route execution mode changed")
    primitive_count = int(config["public_protocol"]["primitive_count"])
    initialization = _validate_initialization_contract(
        _scene_specs(config)[request.scene].get("initialization"),
        primitive_count=primitive_count,
    )
    compiler = _validate_compiler_contract(
        config.get("compiler"),
        primitive_count=primitive_count,
    )
    worldfoam_runtime = _validate_worldfoam_runtime_contract(
        config.get("worldfoam_runtime")
    )
    scene_receipt = {
        **config_receipt["scenes"][request.scene],
        "initialization": initialization,
        "initialization_sha256": _sha256(initialization),
        "compiler": compiler,
        "compiler_sha256": _sha256(compiler),
        "worldfoam_runtime": worldfoam_runtime,
        "worldfoam_runtime_sha256": _sha256(worldfoam_runtime),
    }
    expected_protocol = _repo_path(
        str(scene_receipt["protocol_path"]),
        name="scene protocol",
        must_exist=True,
    )
    if protocol_path != expected_protocol:
        raise ValueError("worker protocol differs from the frozen scene protocol")
    protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
    if (
        protocol.dataset.frame_count != EXPECTED_HELDOUT_FRAMES
        or protocol.final_stage.image_size.as_list() != list(EXPECTED_IMAGE_SIZE)
        or protocol.steps != 300
        or max(stage.frames_per_step for stage in protocol.stages) != 4
        or protocol.target_pixel_budget != EXPECTED_TARGET_PIXELS
        or len(protocol.dataset.heldout_cameras) != 1
    ):
        raise ValueError("worker protocol is not the frozen full-300 all-pixel G4 contract")
    output_path = _repo_path(request.output_path, name="G4 row output")
    if output_path != _expected_output_path(config, request):
        raise ValueError("G4 row output path differs from the frozen matrix")
    work_plan = build_full_pixel_work_plan(protocol, seed=request.seed)
    return (
        config,
        config_receipt,
        protocol,
        route_spec,
        scene_receipt,
        work_plan,
    )


def _validate_bound_file(
    value: Any,
    *,
    name: str,
    split: str,
    cameras: Sequence[str],
) -> tuple[Path, Mapping[str, Any]]:
    receipt = _mapping(value, name=name)
    if set(receipt) != _BOUND_FILE_KEYS:
        raise ValueError(f"{name} keys changed")
    if receipt.get("target_split") != split:
        raise ValueError(f"{name} target split changed")
    if receipt.get("camera_ids") != list(cameras):
        raise ValueError(f"{name} camera ids changed")
    path = _repo_path(str(receipt.get("path", "")), name=name, must_exist=True)
    if (
        receipt.get("bytes") != path.stat().st_size
        or receipt.get("sha256") != file_sha256(path)
    ):
        raise ValueError(f"{name} identity changed")
    return path, receipt


def load_dataset_capability(
    path: Path,
    *,
    request: RowRequest,
    protocol: PaperTrainingProtocol,
    scene_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    capability_path = _repo_path(path, name="dataset capability", must_exist=True)
    payload = _load_json(capability_path)
    if set(payload) != _DATASET_CAPABILITY_KEYS:
        raise ValueError("dataset capability keys changed")
    if (
        payload.get("schema_version") != DATASET_CAPABILITY_SCHEMA_VERSION
        or payload.get("kind") != "worldfoam-public-train-heldout-cache-v1"
        or payload.get("scene") != request.scene
        or payload.get("sample_id") != protocol.dataset.sample_id
        or payload.get("protocol_sha256")
        != file_sha256(
            _repo_path(request.protocol_path, name="G4 protocol", must_exist=True)
        )
        or payload.get("dataset_manifest_sha256")
        != scene_receipt["manifest_sha256"]
        or payload.get("frame_count") != protocol.dataset.frame_count
        or payload.get("image_size") != protocol.final_stage.image_size.as_list()
        or payload.get("train_cameras") != list(protocol.dataset.train_cameras)
        or payload.get("heldout_cameras") != list(protocol.dataset.heldout_cameras)
        or payload.get("public_data") is not True
        or payload.get("calibrated_multiview") is not True
        or payload.get("selected_pixel_reads") is not True
        or payload.get("full_frame_materialization_required") is not False
        or payload.get("initialization_sha256")
        != scene_receipt["initialization_sha256"]
        or payload.get("compiler_sha256") != scene_receipt["compiler_sha256"]
        or payload.get("worldfoam_runtime_sha256")
        != scene_receipt["worldfoam_runtime_sha256"]
    ):
        raise ValueError("dataset capability does not match the frozen public protocol")
    train_path, _ = _validate_bound_file(
        payload.get("train_binding"),
        name="train binding",
        split="train",
        cameras=protocol.dataset.train_cameras,
    )
    heldout_path, _ = _validate_bound_file(
        payload.get("heldout_binding"),
        name="heldout binding",
        split="heldout",
        cameras=protocol.dataset.heldout_cameras,
    )
    factory = _mapping(payload.get("provider_factory"), name="provider factory")
    if set(factory) != _FACTORY_KEYS:
        raise ValueError("provider factory keys changed")
    source_path = _repo_path(
        str(factory.get("source_path", "")),
        name="provider factory source",
        must_exist=True,
    )
    if factory.get("source_sha256") != file_sha256(source_path):
        raise ValueError("provider factory source identity changed")
    if not str(factory.get("module", "")).strip() or not str(
        factory.get("callable", "")
    ).strip():
        raise ValueError("provider factory entrypoint is missing")
    expected_digest = _sha256(
        {key: value for key, value in payload.items() if key != "capability_sha256"}
    )
    if payload.get("capability_sha256") != expected_digest:
        raise ValueError("dataset capability canonical digest changed")

    # This verifier is pure Python and performs the authoritative split,
    # camera-grid, logical-time, decoded-RGB, and cache-byte checks without
    # importing Torch or opening an accelerator runtime.
    from worldfoam_target_dataset_binding import (
        verify_train_heldout_target_dataset_pair,
    )

    pair_receipt = verify_train_heldout_target_dataset_pair(
        train_binding_path=train_path,
        heldout_binding_path=heldout_path,
        required_frame_counts=(protocol.dataset.frame_count,),
    )
    if (
        pair_receipt.get("dataset_id") != protocol.dataset.sample_id
        or pair_receipt.get("train", {}).get("view_ids")
        != list(protocol.dataset.train_cameras)
        or pair_receipt.get("heldout", {}).get("view_ids")
        != list(protocol.dataset.heldout_cameras)
        or pair_receipt.get("common_grid", {}).get("height")
        != protocol.final_stage.image_size.height
        or pair_receipt.get("common_grid", {}).get("width")
        != protocol.final_stage.image_size.width
    ):
        raise ValueError("paired target cache receipt changed the frozen public grid")
    return {**payload, "_verified_pair_receipt": pair_receipt}


def _preflight_state(
    request: RowRequest,
) -> tuple[list[str], dict[str, Any], RowContext | None]:
    """Return blockers without importing Torch, W&B, LPIPS, or native code."""

    blockers: list[str] = []
    details: dict[str, Any] = {"allocation_started": False}
    try:
        (
            config,
            config_receipt,
            protocol,
            route_spec,
            scene_receipt,
            work_plan,
        ) = resolve_row_request(request)
    except Exception as error:
        blockers.append(f"request_contract_invalid:{type(error).__name__}:{error}")
        return blockers, details, None
    details["work_plan"] = work_plan.as_dict()
    details["route_spec"] = dict(route_spec)
    if request.route in {"worldfoam_native4d", "worldfoam_framewise_replay"}:
        tractability = audit_worldfoam_g4_full_schedule(
            protocol=protocol,
            work_plan=work_plan,
            compiler=scene_receipt["compiler"],
            runtime=scene_receipt["worldfoam_runtime"],
        )
        details["worldfoam_full_schedule_tractability"] = tractability.as_dict()
        if tractability.blocker is not None:
            blockers.append(tractability.blocker)
    initialization_blockers, initialization_details = _initialization_blockers(
        scene_receipt["initialization"]
    )
    blockers.extend(initialization_blockers)
    details["initialization"] = {
        **initialization_details,
        "initialization_sha256": scene_receipt["initialization_sha256"],
        "compiler_sha256": scene_receipt["compiler_sha256"],
        "worldfoam_runtime_sha256": scene_receipt["worldfoam_runtime_sha256"],
    }
    if not request.allow_local_mps_execution:
        blockers.append("local_mps_execution_not_acknowledged")
    if sys.platform != "darwin":
        blockers.append("metal_g4_requires_macos")
    if not RUNTIME_CAPABILITY_PATH.is_file():
        blockers.append("runtime_capability_receipt_missing")
    else:
        try:
            runtime = _load_json(RUNTIME_CAPABILITY_PATH)
        except Exception as error:
            blockers.append(f"runtime_capability_receipt_invalid:{type(error).__name__}:{error}")
        else:
            if runtime != REQUIRED_RUNTIME_CAPABILITIES:
                blockers.append("runtime_capabilities_not_verified")
    capability_path = request.dataset_capability_path or default_dataset_capability_path(
        protocol
    )
    try:
        dataset_capability = load_dataset_capability(
            capability_path,
            request=request,
            protocol=protocol,
            scene_receipt=scene_receipt,
        )
    except Exception as error:
        blockers.append(f"public_train_heldout_cache_unavailable:{type(error).__name__}:{error}")
        dataset_capability = None
        try:
            from neural3d_mapped_rgb8_adapter import (
                neural3d_mapped_rgb8_offline_preflight,
            )

            conversion_preflight = neural3d_mapped_rgb8_offline_preflight()
        except Exception as conversion_error:
            blockers.append(
                "bounded_cache_conversion_capability_invalid:"
                f"{type(conversion_error).__name__}:{conversion_error}"
            )
        else:
            details["cache_conversion_preflight"] = conversion_preflight
            if conversion_preflight.get("ready") is not True:
                blockers.append("bounded_cache_conversion_capability_missing")
    details["dataset_capability_path"] = _display(capability_path)
    details["dataset_capability_verified"] = dataset_capability is not None
    executor_module = ROUTE_EXECUTOR_MODULES[request.route]
    try:
        executor_spec = importlib.util.find_spec(executor_module)
    except (ImportError, ModuleNotFoundError, ValueError):
        executor_spec = None
    if executor_spec is None or executor_spec.origin is None:
        blockers.append(f"production_route_executor_missing:{executor_module}")
    else:
        executor_path = Path(executor_spec.origin).resolve()
        if not executor_path.is_file():
            blockers.append(f"production_route_executor_source_missing:{executor_module}")
        else:
            details["executor_source"] = _file_identity(executor_path)
            if request.route.startswith("worldfoam_") and (
                "CERTIFIED_SPATIAL_COMPILE_REUSE = False"
                in executor_path.read_text(encoding="utf-8")
            ):
                blockers.append(
                    "worldfoam_full_schedule_spatial_compile_reuse_unimplemented"
                )
    lpips_status = lpips_alex_asset_status()
    details["lpips_assets"] = lpips_status
    if lpips_status.get("status") != "pass":
        blockers.append("paper_lpips_assets_missing_or_drifted")
    if shutil.which("ffmpeg") is None:
        blockers.append("ffmpeg_media_writer_missing")
    if importlib.util.find_spec("wandb") is None:
        blockers.append("wandb_run_writer_missing")
    try:
        source = _source_identity()
    except Exception as error:
        blockers.append(f"source_identity_unavailable:{type(error).__name__}:{error}")
        source = None
    if source is not None:
        details["source"] = source
        if source.get("repository_dirty") is not False:
            blockers.append("paper_evidence_requires_clean_source")
        if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("repository_commit", ""))):
            blockers.append("source_commit_invalid")
    details["config_receipt"] = config_receipt
    details["protocol"] = protocol.as_dict()
    details["runtime_ready"] = not blockers
    context = None
    if dataset_capability is not None and source is not None:
        context = RowContext(
            request=request,
            config=config,
            config_receipt=config_receipt,
            protocol=protocol,
            route_spec=route_spec,
            scene_receipt=scene_receipt,
            work_plan=work_plan,
            source_commit=str(source["repository_commit"]),
            dataset_capability=dataset_capability,
        )
    return sorted(set(blockers)), details, context


def preflight_blockers(request: RowRequest) -> tuple[list[str], dict[str, Any]]:
    """Return blockers without importing Torch, W&B, LPIPS, or native code."""

    blockers, details, _context = _preflight_state(request)
    return blockers, details


@dataclass(frozen=True)
class PixelChunkPayload:
    """One bounded public target/ray read owned by a dataset provider."""

    target_rgb_f32_cpu: Any
    rays_f32_cpu: Any
    selected_read_receipt: Any


class PublicQualityDataset(Protocol):
    sample_id: str
    train_cameras: tuple[str, ...]
    heldout_cameras: tuple[str, ...]
    frame_count: int
    height: int
    width: int

    def attestation(self) -> Mapping[str, Any]: ...

    def read_train_chunk(self, request: PixelChunkRequest) -> PixelChunkPayload: ...

    def read_heldout_chunk(self, request: PixelChunkRequest) -> PixelChunkPayload: ...

    def close(self) -> None: ...


class WorldFoamTrainingInputsDataset(Protocol):
    """Optional retained-depth seam consumed only by the WorldFoam executor.

    Gaussian route executors remain coupled only to ``PublicQualityDataset``.
    A WorldFoam executor must feature-detect this method and fail closed when
    any of these production objects is unavailable; it must never synthesize a
    procedural world or silently substitute the reduced memory-gate fixture.
    """

    def worldfoam_training_inputs(self) -> Mapping[str, Any]: ...


class CalibratedPublicQualityDataset(Protocol):
    """Optional camera-object seam for STAR/fast-mac production executors."""

    def camera_spec(
        self,
        *,
        split: str,
        camera_index: int,
        frame_index: int,
    ) -> Any: ...


WORLD_FOAM_TRAINING_INPUT_KEYS = frozenset(
    {
        "schema_version",
        "dataset_owner_identity",
        "sample_id",
        "dataset_generation_digest",
        "heldout_dataset_generation_digest",
        "dataset_capability_sha256",
        "initialization_sha256",
        "compiler_sha256",
        "same_representation_group",
        "target_provider",
        "ray_provider",
        "heldout_target_provider",
        "heldout_ray_provider",
        "frame_times",
        "world_initializer",
        "program_factory",
        "background_rgb_f32_cpu",
        "artifact_store_policy",
        "lazy_memory_policy",
        "full_geometry_memory_policy",
        "combined_sgd_policy",
        "maximum_samples_per_launch",
        "maximum_artifact_accounted_bytes_per_entry",
        "cone_tolerance",
        "material_parameterization",
        "material_sgd_policy",
        "maximum_material_state_logical_tensor_bytes",
        "maximum_tracks_per_bundle",
        "maximum_observations_per_bundle",
        "maximum_rows_per_native_block",
        "shared_reverse_mode",
        "input_generation_digest",
    }
)


class ProductionRouteSession(Protocol):
    def begin_step(self, work: StepWork) -> None: ...

    def accumulate_train_chunk(
        self,
        request: PixelChunkRequest,
        payload: PixelChunkPayload,
    ) -> None: ...

    def finish_step(self, work: StepWork) -> None: ...

    def finalize_training(self, checkpoint_path: Path) -> Mapping[str, Any]: ...

    def render_heldout_chunk(
        self,
        request: PixelChunkRequest,
        rays_f32_cpu: Any,
    ) -> Any: ...

    def close(self) -> None: ...


class ProductionRouteExecutor(Protocol):
    def capability(self, context: RowContext) -> Mapping[str, Any]: ...

    def open_session(
        self,
        context: RowContext,
        dataset: PublicQualityDataset,
    ) -> ProductionRouteSession: ...


class HeldoutMediaSink(Protocol):
    def add_frame(self, prediction_hwc: Any, target_hwc: Any) -> None: ...

    def finish(self, *, expected_frame_count: int) -> Path: ...

    def abort(self) -> None: ...


_DATASET_ATTESTATION = {
    "public_data": True,
    "calibrated_multiview": True,
    "procedural_target": False,
    "train_cache_bound": True,
    "heldout_cache_bound": True,
    "selected_pixel_reads": True,
    "full_frame_materialization_required": False,
}
_EXECUTOR_CAPABILITY_KEYS = {
    "schema_version",
    "route",
    "lane",
    "execution_mode",
    "backend",
    "real_native",
    "native_extension_attested",
    "fake_native",
    "source_only",
    "procedural_target",
    "public_target_provider",
    "heldout_evaluator",
    "full_geometry_trainable",
    "compiled_shared_adjoint",
    "same_representation_framewise_replay",
    "proxy_or_test_artifact",
    "measurement_is_simulated",
}
_TRAINING_RECEIPT_KEYS = {
    "optimizer_steps",
    "target_pixels_consumed",
    "sampled_image_count",
    "pixel_chunk_count",
    "rasterized_pixels",
    "parameter_count",
    "parameter_bytes",
    "process_lifetime_peak_rss_through_checkpoint_bytes",
    "sampled_peak_mps_driver_during_training_and_checkpoint_bytes",
    "training_and_checkpoint_elapsed_s",
    "representation_sha256",
    "checkpoint_step",
}
_ROW_MEASUREMENT_KEYS = {
    "process_lifetime_peak_rss_through_heldout_evaluation_bytes",
    "sampled_peak_mps_driver_through_heldout_evaluation_bytes",
    "executor_dataset_and_model_setup_elapsed_s",
    "heldout_evaluation_elapsed_s",
    "full_row_through_heldout_evaluation_elapsed_s",
}


def _fields(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    fields = getattr(value, "__dict__", None)
    if not isinstance(fields, Mapping):
        raise TypeError(f"{name} must expose mapping or dataclass fields")
    return fields


def _validate_dataset(
    dataset: PublicQualityDataset,
    *,
    context: RowContext,
) -> None:
    protocol = context.protocol
    exact = {
        "sample_id": protocol.dataset.sample_id,
        "train_cameras": tuple(protocol.dataset.train_cameras),
        "heldout_cameras": tuple(protocol.dataset.heldout_cameras),
        "frame_count": protocol.dataset.frame_count,
        "height": protocol.final_stage.image_size.height,
        "width": protocol.final_stage.image_size.width,
    }
    drift = [key for key, expected in exact.items() if getattr(dataset, key, None) != expected]
    if drift:
        raise ValueError("public dataset provider drifted: " + ", ".join(drift))
    attestation = _mapping(dataset.attestation(), name="dataset attestation")
    if dict(attestation) != _DATASET_ATTESTATION:
        raise ValueError("public dataset provider attestation is not production eligible")


def validate_worldfoam_training_inputs(
    dataset: PublicQualityDataset,
    *,
    context: RowContext,
) -> dict[str, Any]:
    accessor = getattr(dataset, "worldfoam_training_inputs", None)
    if not callable(accessor):
        raise TypeError(
            "WorldFoam public dataset lacks sealed worldfoam_training_inputs()"
        )
    sealed = accessor()
    inputs = dict(_mapping(sealed, name="WorldFoam training inputs"))
    from worldfoam_public_quality_inputs import WorldFoamPublicQualityInputs

    if set(WORLD_FOAM_TRAINING_INPUT_KEYS) != set(
        WorldFoamPublicQualityInputs.FIELD_NAMES
    ):
        raise RuntimeError(
            "row-worker and neutral WorldFoam training-input schemas differ"
        )
    if set(inputs) != set(WORLD_FOAM_TRAINING_INPUT_KEYS):
        raise ValueError("WorldFoam training-input keys changed")
    assertion = getattr(sealed, "assert_current", None)
    if not callable(assertion):
        raise TypeError("WorldFoam training inputs are not cold sealed")
    assertion(dataset=dataset, context=context)
    if inputs["schema_version"] != 1 or inputs["sample_id"] != context.protocol.dataset.sample_id:
        raise ValueError("WorldFoam training-input schema/sample changed")
    exact_digests = {
        "dataset_capability_sha256": context.dataset_capability[
            "capability_sha256"
        ],
        "initialization_sha256": context.scene_receipt["initialization_sha256"],
        "compiler_sha256": context.scene_receipt["compiler_sha256"],
    }
    for key, expected in exact_digests.items():
        if inputs.get(key) != expected:
            raise ValueError(f"WorldFoam training input {key} changed")
    for key in (
        "dataset_generation_digest",
        "heldout_dataset_generation_digest",
        "input_generation_digest",
    ):
        if not _valid_sha256(inputs.get(key)):
            raise ValueError(f"WorldFoam training input {key} is invalid")
    if inputs.get("same_representation_group") != context.route_spec.get(
        "same_representation_group"
    ):
        raise ValueError("WorldFoam same-representation group changed")
    for key in (
        "target_provider",
        "ray_provider",
        "heldout_target_provider",
        "heldout_ray_provider",
        "world_initializer",
        "program_factory",
        "background_rgb_f32_cpu",
        "artifact_store_policy",
        "lazy_memory_policy",
        "full_geometry_memory_policy",
        "combined_sgd_policy",
        "material_parameterization",
        "material_sgd_policy",
    ):
        if inputs.get(key) is None:
            raise ValueError(f"WorldFoam training input {key} is missing")
    frame_times = inputs.get("frame_times")
    if (
        not isinstance(frame_times, tuple)
        or len(frame_times) != context.protocol.dataset.frame_count
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in frame_times
        )
    ):
        raise ValueError("WorldFoam training frame_times changed")
    for name, provider, view_count in (
        (
            "target_provider",
            inputs["target_provider"],
            len(context.protocol.dataset.train_cameras),
        ),
        (
            "ray_provider",
            inputs["ray_provider"],
            len(context.protocol.dataset.train_cameras),
        ),
        (
            "heldout_target_provider",
            inputs["heldout_target_provider"],
            len(context.protocol.dataset.heldout_cameras),
        ),
        (
            "heldout_ray_provider",
            inputs["heldout_ray_provider"],
            len(context.protocol.dataset.heldout_cameras),
        ),
    ):
        expected_provider_shape = {
            "view_count": view_count,
            "frame_count": context.protocol.dataset.frame_count,
            "height": context.protocol.final_stage.image_size.height,
            "width": context.protocol.final_stage.image_size.width,
        }
        drift = [
            key
            for key, expected in expected_provider_shape.items()
            if getattr(provider, key, None) != expected
        ]
        if drift:
            raise ValueError(f"WorldFoam {name} grid changed: {', '.join(drift)}")
    import torch

    background = inputs["background_rgb_f32_cpu"]
    if (
        not isinstance(background, torch.Tensor)
        or background.device.type != "cpu"
        or background.dtype != torch.float32
        or tuple(background.shape) != (3,)
        or not background.is_contiguous()
        or not bool(torch.isfinite(background).all().item())
        or background.tolist()
        != list(context.scene_receipt["worldfoam_runtime"]["background_rgb"])
    ):
        raise ValueError("WorldFoam background tensor changed")
    runtime = context.scene_receipt["worldfoam_runtime"]
    for key in (
        "maximum_samples_per_launch",
        "maximum_artifact_accounted_bytes_per_entry",
        "maximum_material_state_logical_tensor_bytes",
        "maximum_tracks_per_bundle",
        "maximum_observations_per_bundle",
        "maximum_rows_per_native_block",
        "cone_tolerance",
        "shared_reverse_mode",
    ):
        if inputs.get(key) != runtime[key]:
            raise ValueError(f"WorldFoam runtime input {key} changed")
    for key in (
        "artifact_store_policy",
        "lazy_memory_policy",
        "full_geometry_memory_policy",
        "combined_sgd_policy",
        "material_parameterization",
        "material_sgd_policy",
    ):
        policy_fields = getattr(inputs[key], "__dict__", None)
        if not isinstance(policy_fields, Mapping):
            raise TypeError(f"WorldFoam {key} does not expose immutable fields")
        if dict(policy_fields) != dict(runtime[key]):
            raise ValueError(f"WorldFoam runtime policy {key} changed")
        assertion = getattr(inputs[key], "assert_valid", None)
        if callable(assertion):
            if key == "full_geometry_memory_policy":
                assertion(reverse_mode=runtime["shared_reverse_mode"])
            else:
                assertion()
        elif key != "artifact_store_policy":
            raise TypeError(f"WorldFoam {key} is not a validated policy object")
    return inputs


def _validate_executor_capability(
    capability: Mapping[str, Any],
    *,
    context: RowContext,
) -> dict[str, Any]:
    value = dict(_mapping(capability, name="route executor capability"))
    if set(value) != _EXECUTOR_CAPABILITY_KEYS:
        raise ValueError("route executor capability keys changed")
    expected = {
        "schema_version": 1,
        "route": context.request.route,
        "lane": context.route_spec["lane"],
        "execution_mode": context.route_spec["execution_mode"],
        "backend": context.route_spec["backend"],
        "real_native": True,
        "native_extension_attested": False,
        "fake_native": False,
        "source_only": False,
        "procedural_target": False,
        "public_target_provider": True,
        "heldout_evaluator": True,
        "full_geometry_trainable": True,
        "compiled_shared_adjoint": context.request.route == "worldfoam_native4d",
        "same_representation_framewise_replay": (
            context.request.route == "worldfoam_framewise_replay"
        ),
        "proxy_or_test_artifact": False,
        "measurement_is_simulated": False,
    }
    if value != expected:
        raise ValueError("route executor is not the exact real production capability")
    return value


def _validate_pixel_payload(
    payload: PixelChunkPayload,
    *,
    request: PixelChunkRequest,
) -> PixelChunkPayload:
    import torch

    if not isinstance(payload, PixelChunkPayload):
        raise TypeError("dataset must return PixelChunkPayload")
    target = payload.target_rgb_f32_cpu
    rays = payload.rays_f32_cpu
    if (
        not isinstance(target, torch.Tensor)
        or target.device.type != "cpu"
        or target.dtype != torch.float32
        or tuple(target.shape) != (request.pixel_count, 3)
        or not target.is_contiguous()
        or not bool(torch.isfinite(target).all().item())
        or float(target.min().item()) < 0.0
        or float(target.max().item()) > 1.0
    ):
        raise ValueError("public target chunk violated the bounded CPU RGB contract")
    if (
        not isinstance(rays, torch.Tensor)
        or rays.device.type != "cpu"
        or rays.dtype != torch.float32
        or tuple(rays.shape) != (request.pixel_count, 6)
        or not rays.is_contiguous()
        or not bool(torch.isfinite(rays).all().item())
    ):
        raise ValueError("public ray chunk violated the bounded CPU ray contract")
    receipt = _fields(payload.selected_read_receipt, name="selected read receipt")
    if (
        receipt.get("observation_count") != request.pixel_count
        or receipt.get("selection_mode")
        not in {"direct_pixels", "certified_bounded_region"}
        or receipt.get("full_frame_materialization_count") != 0
        or receipt.get("maximum_full_frame_materialization_tensor_bytes") != 0
    ):
        raise ValueError("public target chunk used an ineligible materialization path")
    if request.pixel_ids is not None and receipt.get(
        "requested_pixel_ids_sha256"
    ) != _sha256(request.pixel_ids):
        raise ValueError("public target chunk is not bound to the requested sensor pixels")
    return payload


def validate_training_receipt(
    receipt: Mapping[str, Any],
    *,
    context: RowContext,
    checkpoint_path: Path,
) -> dict[str, Any]:
    value = dict(_mapping(receipt, name="training receipt"))
    if set(value) != _TRAINING_RECEIPT_KEYS:
        raise ValueError("training receipt keys changed")
    exact = {
        "optimizer_steps": context.protocol.steps,
        "target_pixels_consumed": context.work_plan.target_pixels,
        "sampled_image_count": context.work_plan.sampled_image_count,
        "pixel_chunk_count": context.work_plan.pixel_chunk_count,
        "checkpoint_step": context.protocol.steps,
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise ValueError(f"training receipt {key} changed; reduced-pixel rows are forbidden")
    for key in (
        "rasterized_pixels",
        "parameter_count",
        "parameter_bytes",
        "process_lifetime_peak_rss_through_checkpoint_bytes",
        "sampled_peak_mps_driver_during_training_and_checkpoint_bytes",
    ):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"training receipt {key} must be a nonnegative integer")
    if value["parameter_count"] < 1 or value["parameter_bytes"] < 1:
        raise ValueError("training receipt contains no trainable representation")
    elapsed = value.get("training_and_checkpoint_elapsed_s")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0.0
    ):
        raise ValueError(
            "training receipt training_and_checkpoint_elapsed_s must be finite and positive"
        )
    if not _valid_sha256(value.get("representation_sha256")):
        raise ValueError("training receipt representation digest is invalid")
    checkpoint = _repo_path(checkpoint_path, name="final checkpoint", must_exist=True)
    if checkpoint.stat().st_size < 1:
        raise ValueError("final checkpoint is empty")
    return value


def run_training_lifecycle(
    context: RowContext,
    *,
    dataset: PublicQualityDataset,
    session: ProductionRouteSession,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Execute the frozen schedule and visit its exact target-pixel stream."""

    consumed_chunks = 0
    consumed_pixels = 0
    consumed_images: set[tuple[int, int]] = set()
    for work in context.work_plan.steps:
        session.begin_step(work)
        step_chunks = 0
        for request in context.work_plan.iter_step_training_chunks(work):
            if request.step != work.step:
                raise ArithmeticError("training chunk stream crossed a step boundary")
            payload = _validate_pixel_payload(
                dataset.read_train_chunk(request),
                request=request,
            )
            session.accumulate_train_chunk(request, payload)
            consumed_chunks += 1
            step_chunks += 1
            consumed_pixels += request.pixel_count
            consumed_images.add((int(request.step), int(request.sample_slot)))
            del payload
        if step_chunks < 1:
            raise ArithmeticError("training step emitted no target-pixel chunks")
        session.finish_step(work)
    if (
        consumed_chunks != context.work_plan.pixel_chunk_count
        or consumed_pixels != context.work_plan.target_pixels
        or len(consumed_images) != context.work_plan.sampled_image_count
    ):
        raise ArithmeticError("training lifecycle did not consume the sealed work plan")
    receipt = session.finalize_training(checkpoint_path)
    return validate_training_receipt(
        receipt,
        context=context,
        checkpoint_path=checkpoint_path,
    )


@dataclass(frozen=True)
class HeldoutEvaluationReceipt:
    metrics: dict[str, float]
    frame_count: int
    pixel_count: int
    pixel_chunk_count: int
    coverage_sha256: str
    media_path: Path


class StreamingSideBySideMp4Sink:
    """Write target/prediction media one frame at a time through ffmpeg."""

    def __init__(self, path: Path, *, height: int, width: int, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for G4 heldout media")
        self.path = _repo_path(path, name="heldout media")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        self.height = int(height)
        self.width = int(width)
        self.frame_count = 0
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{2 * self.width}x{self.height}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            "-an",
            "-vf",
            "format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-tag:v",
            "avc1",
            "-movflags",
            "+faststart",
            str(self.partial),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def add_frame(self, prediction_hwc: Any, target_hwc: Any) -> None:
        import torch

        if self.process.stdin is None:
            raise RuntimeError("heldout media writer stdin is closed")
        if (
            not isinstance(prediction_hwc, torch.Tensor)
            or not isinstance(target_hwc, torch.Tensor)
            or tuple(prediction_hwc.shape) != (self.height, self.width, 3)
            or tuple(target_hwc.shape) != (self.height, self.width, 3)
        ):
            raise ValueError("heldout media frame shape changed")
        frame = torch.cat(
            [target_hwc.detach().cpu(), prediction_hwc.detach().cpu()], dim=1
        )
        rgb8 = frame.clamp(0.0, 1.0).mul(255.0).to(torch.uint8).contiguous()
        self.process.stdin.write(memoryview(rgb8.numpy()).cast("B"))
        self.frame_count += 1

    def finish(self, *, expected_frame_count: int) -> Path:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            self.abort()
            raise RuntimeError("ffmpeg failed while writing full-temporal G4 media")
        if self.frame_count != int(expected_frame_count):
            self.abort()
            raise ArithmeticError("heldout media frame coverage changed")
        if not self.partial.is_file() or self.partial.stat().st_size < 1:
            self.abort()
            raise FileNotFoundError("ffmpeg emitted no heldout media")
        os.replace(self.partial, self.path)
        return self.path

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self.partial.unlink(missing_ok=True)


def evaluate_final_checkpoint(
    context: RowContext,
    *,
    dataset: PublicQualityDataset,
    session: ProductionRouteSession,
    media_sink: HeldoutMediaSink,
    lpips_metric: Callable[[Any, Any], float] | None = None,
) -> HeldoutEvaluationReceipt:
    """Evaluate all heldout pixels, reconstructing only one frame at a time."""

    import torch
    from perceptual_metrics import video_lpips

    metric = video_lpips if lpips_metric is None else lpips_metric
    protocol = context.protocol
    height = protocol.final_stage.image_size.height
    width = protocol.final_stage.image_size.width
    pixels = height * width
    accumulator = PaperRGBMetricAccumulator()
    coverage = hashlib.sha256()
    lpips_sum = 0.0
    frame_count = 0
    pixel_count = 0
    chunk_count = 0
    heldout_maximum_pixels_per_chunk = int(
        getattr(
            context.work_plan,
            "heldout_maximum_pixels_per_chunk",
            context.work_plan.maximum_pixels_per_chunk,
        )
    )
    if heldout_maximum_pixels_per_chunk < 1:
        raise ValueError("heldout pixel chunk bound must be positive")
    try:
        for camera_index, _camera in enumerate(protocol.dataset.heldout_cameras):
            for frame_index in range(protocol.dataset.frame_count):
                prediction_frame = torch.empty((pixels, 3), dtype=torch.float32)
                target_frame = torch.empty((pixels, 3), dtype=torch.float32)
                covered = 0
                for pixel_start in range(
                    0, pixels, heldout_maximum_pixels_per_chunk
                ):
                    request = PixelChunkRequest(
                        split="heldout",
                        step=None,
                        sample_slot=None,
                        camera_index=camera_index,
                        frame_index=frame_index,
                        pixel_start=pixel_start,
                        pixel_count=min(
                            heldout_maximum_pixels_per_chunk,
                            pixels - pixel_start,
                        ),
                        image_height=height,
                        image_width=width,
                    )
                    payload = _validate_pixel_payload(
                        dataset.read_heldout_chunk(request),
                        request=request,
                    )
                    prediction = session.render_heldout_chunk(
                        request,
                        payload.rays_f32_cpu,
                    )
                    if (
                        not isinstance(prediction, torch.Tensor)
                        or tuple(prediction.shape) != (request.pixel_count, 3)
                        or not bool(torch.isfinite(prediction).all().item())
                    ):
                        raise ValueError("heldout renderer returned an invalid RGB chunk")
                    prediction_cpu = (
                        prediction.detach().to(device="cpu", dtype=torch.float32).contiguous()
                    )
                    target_frame[request.pixel_start : request.pixel_stop].copy_(
                        payload.target_rgb_f32_cpu
                    )
                    prediction_frame[
                        request.pixel_start : request.pixel_stop
                    ].copy_(prediction_cpu)
                    coverage.update(_canonical_bytes(request.as_dict()))
                    coverage.update(b"\n")
                    covered += request.pixel_count
                    pixel_count += request.pixel_count
                    chunk_count += 1
                    del payload, prediction, prediction_cpu
                if covered != pixels:
                    raise ArithmeticError("heldout frame was not reconstructed exactly")
                prediction_hwc = prediction_frame.reshape(height, width, 3)
                target_hwc = target_frame.reshape(height, width, 3)
                accumulator.update(
                    prediction_hwc.unsqueeze(0),
                    target_hwc.unsqueeze(0),
                )
                lpips_value = float(
                    metric(
                        prediction_hwc.unsqueeze(0),
                        target_hwc.unsqueeze(0),
                    )
                )
                if not math.isfinite(lpips_value) or lpips_value < 0.0:
                    raise ValueError("heldout LPIPS is non-finite or negative")
                lpips_sum += lpips_value
                media_sink.add_frame(prediction_hwc, target_hwc)
                frame_count += 1
                del prediction_frame, target_frame, prediction_hwc, target_hwc
        expected_frames = (
            len(protocol.dataset.heldout_cameras) * protocol.dataset.frame_count
        )
        if frame_count != expected_frames:
            raise ArithmeticError("heldout evaluator did not cover the full temporal set")
        media_path = media_sink.finish(expected_frame_count=expected_frames)
    except Exception:
        media_sink.abort()
        raise
    metrics = accumulator.metrics(prefix="heldout_eval")
    result_metrics = {
        "heldout_eval_psnr": float(metrics["heldout_eval_psnr"]),
        "heldout_eval_ssim": float(metrics["heldout_eval_ssim"]),
        "heldout_eval_lpips": lpips_sum / float(frame_count),
        "heldout_eval_l1": float(metrics["heldout_eval_l1"]),
    }
    if set(result_metrics) != set(REQUIRED_METRICS):
        raise ArithmeticError("heldout metric key set changed")
    return HeldoutEvaluationReceipt(
        metrics=result_metrics,
        frame_count=frame_count,
        pixel_count=pixel_count,
        pixel_chunk_count=chunk_count,
        coverage_sha256=coverage.hexdigest(),
        media_path=media_path,
    )


def _load_dataset_provider(context: RowContext) -> PublicQualityDataset:
    factory_receipt = _mapping(
        context.dataset_capability["provider_factory"],
        name="provider factory",
    )
    module = importlib.import_module(str(factory_receipt["module"]))
    module_path = Path(str(getattr(module, "__file__", ""))).resolve()
    expected_path = _repo_path(
        str(factory_receipt["source_path"]),
        name="provider factory source",
        must_exist=True,
    )
    if module_path != expected_path or file_sha256(module_path) != factory_receipt[
        "source_sha256"
    ]:
        raise ValueError("loaded public dataset provider differs from its capability")
    factory = getattr(module, str(factory_receipt["callable"]), None)
    if not callable(factory):
        raise AttributeError("public dataset provider factory callable is missing")
    dataset = factory(context=context, capability=context.dataset_capability)
    _validate_dataset(dataset, context=context)
    return dataset


def load_public_quality_dataset(context: RowContext) -> PublicQualityDataset:
    """Load the already-verified mapped public provider for one sealed context."""

    return _load_dataset_provider(context)


def _load_route_executor(context: RowContext) -> ProductionRouteExecutor:
    module_name = ROUTE_EXECUTOR_MODULES[context.request.route]
    module = importlib.import_module(module_name)
    module_path = _repo_path(
        Path(str(getattr(module, "__file__", ""))).resolve(),
        name="route executor source",
        must_exist=True,
    )
    factory = getattr(module, "create_public_quality_executor", None)
    if not callable(factory):
        raise AttributeError(
            f"{module_name} must export create_public_quality_executor(*, context)"
        )
    executor = factory(context=context)
    capability_method = getattr(executor, "capability", None)
    open_session = getattr(executor, "open_session", None)
    if not callable(capability_method) or not callable(open_session):
        raise TypeError("production route executor protocol is incomplete")
    _validate_executor_capability(capability_method(context), context=context)
    return executor


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = _repo_path(path, name="JSON artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    encoded = (
        json.dumps(
            serialize_config_value(dict(payload)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_wandb_run_file(
    context: RowContext,
    *,
    metrics: Mapping[str, float],
    cost: Mapping[str, int | float],
    media_path: Path,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"online", "offline"}:
        raise ValueError("G4 W&B mode must be online or offline")
    import wandb

    run_prefix = (
        "g4v2-selected"
        if getattr(context.work_plan, "workload_receipt", None) is not None
        else "g4"
    )
    run_id = (
        f"{run_prefix}-{context.request.scene}-{context.request.seed}-"
        f"{context.request.route}"
    )
    wandb_root = context.request.output_path.parent / "wandb_state"
    wandb_root.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project="dynaworld",
        name=run_id,
        id=run_id,
        resume="never",
        reinit="finish_previous",
        mode=mode,
        dir=str(wandb_root),
        tags=[
            "worldfoam-g4-public-quality-v1",
            context.request.scene,
            context.request.route,
            f"seed-{context.request.seed}",
            "full-300-heldout",
        ],
        config={
            "row_id": (
                f"{context.request.scene}/seed_{context.request.seed}/"
                f"{context.request.route}"
            ),
            "source_commit": context.source_commit,
            "protocol_sha256": context.scene_receipt["protocol_sha256"],
            "manifest_sha256": context.scene_receipt["manifest_sha256"],
            "sample_schedule_sha256": context.work_plan.sample_schedule_sha256,
            "evaluator_sha256": paper_evaluator_contract()["sha256"],
            "initialization_sha256": context.scene_receipt[
                "initialization_sha256"
            ],
            "compiler_sha256": context.scene_receipt["compiler_sha256"],
        },
        settings=wandb.Settings(disable_git=True, disable_code=True),
    )
    try:
        run.log(
            {
                **{f"heldout/{key.removeprefix('heldout_eval_')}": value for key, value in metrics.items()},
                **{f"cost/{key}": value for key, value in cost.items()},
                "heldout/full_temporal_side_by_side": wandb.Video(
                    str(media_path), format="mp4"
                ),
            },
            step=context.protocol.steps,
        )
        run_dir = Path(str(run.dir)).resolve()
        actual_run_id = str(run.id)
    finally:
        run.finish()
    run_file = run_dir.parent / f"run-{actual_run_id}.wandb"
    if not run_file.is_file() or run_file.stat().st_size < 1:
        raise FileNotFoundError(
            f"W&B did not finalize its exact run file: {run_file}"
        )
    return _file_identity(
        run_file,
        run_id=actual_run_id,
        mode=mode,
    )


def assemble_raw_row(
    context: RowContext,
    *,
    executor_capability: Mapping[str, Any],
    training: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    evaluation: HeldoutEvaluationReceipt,
    heldout_media: Mapping[str, Any],
    wandb_run_file: Mapping[str, Any],
    row_measurements: Mapping[str, Any],
) -> dict[str, Any]:
    public = context.config["public_protocol"]
    if set(row_measurements) != _ROW_MEASUREMENT_KEYS:
        raise ValueError("full-row measurement keys changed")
    for key in (
        "process_lifetime_peak_rss_through_heldout_evaluation_bytes",
        "sampled_peak_mps_driver_through_heldout_evaluation_bytes",
    ):
        value = row_measurements[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"full-row measurement {key} must be nonnegative integer")
    for key in (
        "executor_dataset_and_model_setup_elapsed_s",
        "heldout_evaluation_elapsed_s",
        "full_row_through_heldout_evaluation_elapsed_s",
    ):
        value = row_measurements[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"full-row measurement {key} must be finite and positive")
    cost = {
        "optimizer_steps": int(training["optimizer_steps"]),
        "target_pixels": int(training["target_pixels_consumed"]),
        "rasterized_pixels": int(training["rasterized_pixels"]),
        "parameter_count": int(training["parameter_count"]),
        "parameter_bytes": int(training["parameter_bytes"]),
        "serialized_checkpoint_bytes": int(checkpoint["bytes"]),
        "final_active_primitive_count_per_render": int(public["primitive_count"]),
        "stored_primitive_state_count": int(public["primitive_count"])
        * (
            int(context.protocol.dataset.frame_count)
            if context.request.route == "dynamic_3dgs"
            else 1
        ),
        "process_lifetime_peak_rss_through_checkpoint_bytes": int(
            training["process_lifetime_peak_rss_through_checkpoint_bytes"]
        ),
        "sampled_peak_mps_driver_during_training_and_checkpoint_bytes": int(
            training[
                "sampled_peak_mps_driver_during_training_and_checkpoint_bytes"
            ]
        ),
        "training_and_checkpoint_elapsed_s": float(
            training["training_and_checkpoint_elapsed_s"]
        ),
        "process_lifetime_peak_rss_through_heldout_evaluation_bytes": int(
            row_measurements[
                "process_lifetime_peak_rss_through_heldout_evaluation_bytes"
            ]
        ),
        "sampled_peak_mps_driver_through_heldout_evaluation_bytes": int(
            row_measurements[
                "sampled_peak_mps_driver_through_heldout_evaluation_bytes"
            ]
        ),
        "executor_dataset_and_model_setup_elapsed_s": float(
            row_measurements["executor_dataset_and_model_setup_elapsed_s"]
        ),
        "heldout_evaluation_elapsed_s": float(
            row_measurements["heldout_evaluation_elapsed_s"]
        ),
        "full_row_through_heldout_evaluation_elapsed_s": float(
            row_measurements["full_row_through_heldout_evaluation_elapsed_s"]
        ),
    }
    if set(cost) != set(REQUIRED_COST):
        raise ArithmeticError("G4 cost key set changed")
    route_attestation = {
        key: executor_capability[key]
        for key in (
            "real_native",
            "native_extension_attested",
            "fake_native",
            "source_only",
            "procedural_target",
            "public_target_provider",
            "heldout_evaluator",
            "full_geometry_trainable",
            "compiled_shared_adjoint",
            "same_representation_framewise_replay",
        )
    }
    row = {
        "schema_version": ROW_SCHEMA_VERSION,
        "row_kind": ROW_KIND,
        "row_id": (
            f"{context.request.scene}/seed_{context.request.seed}/"
            f"{context.request.route}"
        ),
        "scene": context.request.scene,
        "seed": context.request.seed,
        "route": context.request.route,
        "lane": context.route_spec["lane"],
        "execution_mode": context.route_spec["execution_mode"],
        "backend": context.route_spec["backend"],
        "protocol_path": context.scene_receipt["protocol_path"],
        "protocol_sha256": context.scene_receipt["protocol_sha256"],
        "dataset_manifest_path": context.scene_receipt["manifest_path"],
        "dataset_manifest_sha256": context.scene_receipt["manifest_sha256"],
        "sample_id": context.protocol.dataset.sample_id,
        "train_cameras": list(context.protocol.dataset.train_cameras),
        "heldout_cameras": list(context.protocol.dataset.heldout_cameras),
        "frame_count": context.protocol.dataset.frame_count,
        "image_size": context.protocol.final_stage.image_size.as_list(),
        "optimizer_steps": context.protocol.steps,
        "frames_per_step": int(public["frames_per_step"]),
        "primitive_state_temporal_scope": (
            "per_frame" if context.request.route == "dynamic_3dgs" else "shared_across_time"
        ),
        "target_pixel_budget": context.work_plan.target_pixels,
        "sample_schedule_sha256": context.work_plan.sample_schedule_sha256,
        "evaluator_sha256": paper_evaluator_contract()["sha256"],
        "representation_sha256": training["representation_sha256"],
        "source_commit": context.source_commit,
        "source_dirty": False,
        "public_quality_evidence": True,
        "paper_evidence_eligible": True,
        "proxy_or_test_artifact": False,
        "measurement_is_simulated": False,
        "smoke": False,
        "dataset_is_public": True,
        "calibrated_multiview": True,
        "final_checkpoint_evaluation": True,
        "full_temporal_heldout_evaluation": True,
        "route_attestation": route_attestation,
        "checkpoint": dict(checkpoint),
        "heldout_media": dict(heldout_media),
        "wandb_run_file": dict(wandb_run_file),
        "metrics": dict(evaluation.metrics),
        "cost": cost,
    }
    if set(row) != set(ROW_KEYS):
        raise ArithmeticError("raw G4 row key set changed")
    return row


def _verify_and_publish_raw_row(context: RowContext, row: Mapping[str, Any]) -> Path:
    from verify_worldfoam_public_quality_ablation import _validate_row

    output = _repo_path(context.request.output_path, name="G4 row output")
    if output.exists():
        raise FileExistsError(f"G4 row already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.verify")
    temporary.unlink(missing_ok=True)
    try:
        _atomic_write_json(temporary, row)
        with_receipt = {
            **dict(row),
            "receipt": _file_identity(temporary),
        }
        errors = _validate_row(
            with_receipt,
            config=context.config,
            config_receipt=context.config_receipt,
            artifact_source_commit=context.source_commit,
        )
        if errors:
            raise ValueError(
                "raw G4 row failed independent validation: " + "; ".join(errors)
            )
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def execute_row_lifecycle(
    request: RowRequest,
    *,
    wandb_mode: str = "offline",
) -> dict[str, Any]:
    """Run one real row; no output row is published until every gate passes."""

    blockers, details, context = _preflight_state(request)
    if blockers or context is None:
        raise RuntimeError(
            "G4 row aborted before allocation: " + ", ".join(blockers)
        )
    dataset: PublicQualityDataset | None = None
    session: ProductionRouteSession | None = None
    media_sink: HeldoutMediaSink | None = None
    row_memory_sampler: Any | None = None
    try:
        row_started_at = time.perf_counter()
        import torch
        from device_memory import DeviceMemorySampler

        setup_started_at = time.perf_counter()
        row_memory_sampler = DeviceMemorySampler(torch.device("mps"))
        row_memory_sampler.start()
        dataset = _load_dataset_provider(context)
        executor = _load_route_executor(context)
        executor_capability = _validate_executor_capability(
            executor.capability(context),
            context=context,
        )
        if request.route.startswith("worldfoam_"):
            validate_worldfoam_training_inputs(dataset, context=context)
        session = executor.open_session(context, dataset)
        setup_elapsed_s = time.perf_counter() - setup_started_at
        checkpoint_path = request.output_path.parent / "checkpoint_final.pt"
        training = run_training_lifecycle(
            context,
            dataset=dataset,
            session=session,
            checkpoint_path=checkpoint_path,
        )
        checkpoint = _file_identity(
            checkpoint_path,
            step=context.protocol.steps,
        )
        media_sink = StreamingSideBySideMp4Sink(
            request.output_path.parent / "heldout_full_temporal.mp4",
            height=context.protocol.final_stage.image_size.height,
            width=context.protocol.final_stage.image_size.width,
            fps=context.protocol.dataset.fps,
        )
        evaluation_started_at = time.perf_counter()
        evaluation = evaluate_final_checkpoint(
            context,
            dataset=dataset,
            session=session,
            media_sink=media_sink,
        )
        heldout_evaluation_elapsed_s = time.perf_counter() - evaluation_started_at
        media_sink = None
        expected_eval_pixels = (
            len(context.protocol.dataset.heldout_cameras)
            * context.protocol.dataset.frame_count
            * context.protocol.final_stage.image_size.pixels
        )
        if evaluation.pixel_count != expected_eval_pixels:
            raise ArithmeticError("G4 heldout evaluator changed full-pixel coverage")
        row_memory_sampler.stop()
        full_row_memory = row_memory_sampler.stats()
        full_row_elapsed_s = time.perf_counter() - row_started_at
        row_measurements = {
            "process_lifetime_peak_rss_through_heldout_evaluation_bytes": (
                _process_lifetime_peak_rss_bytes()
            ),
            "sampled_peak_mps_driver_through_heldout_evaluation_bytes": int(
                max(
                    int(full_row_memory["sampled_peak_driver_allocated_bytes"]),
                    int(
                        training[
                            "sampled_peak_mps_driver_during_training_and_checkpoint_bytes"
                        ]
                    ),
                )
            ),
            "executor_dataset_and_model_setup_elapsed_s": float(setup_elapsed_s),
            "heldout_evaluation_elapsed_s": float(heldout_evaluation_elapsed_s),
            "full_row_through_heldout_evaluation_elapsed_s": float(
                full_row_elapsed_s
            ),
        }
        heldout_media = _file_identity(
            evaluation.media_path,
            camera_ids=list(context.protocol.dataset.heldout_cameras),
            frame_count=evaluation.frame_count,
        )
        session.close()
        session = None
        dataset.close()
        dataset = None
        preliminary_cost = {
            "optimizer_steps": int(training["optimizer_steps"]),
            "target_pixels": int(training["target_pixels_consumed"]),
            "rasterized_pixels": int(training["rasterized_pixels"]),
            "parameter_count": int(training["parameter_count"]),
            "parameter_bytes": int(training["parameter_bytes"]),
            "serialized_checkpoint_bytes": int(checkpoint["bytes"]),
            "final_active_primitive_count_per_render": int(
                context.config["public_protocol"]["primitive_count"]
            ),
            "stored_primitive_state_count": int(
                context.config["public_protocol"]["primitive_count"]
            )
            * (
                int(context.protocol.dataset.frame_count)
                if context.request.route == "dynamic_3dgs"
                else 1
            ),
            "process_lifetime_peak_rss_through_checkpoint_bytes": int(
                training["process_lifetime_peak_rss_through_checkpoint_bytes"]
            ),
            "sampled_peak_mps_driver_during_training_and_checkpoint_bytes": int(
                training[
                    "sampled_peak_mps_driver_during_training_and_checkpoint_bytes"
                ]
            ),
            "training_and_checkpoint_elapsed_s": float(
                training["training_and_checkpoint_elapsed_s"]
            ),
            **row_measurements,
        }
        wandb_receipt = write_wandb_run_file(
            context,
            metrics=evaluation.metrics,
            cost=preliminary_cost,
            media_path=evaluation.media_path,
            mode=wandb_mode,
        )
        if _source_identity() != {
            "repository_commit": context.source_commit,
            "repository_dirty": False,
        }:
            raise RuntimeError("source changed while executing the G4 row")
        row = assemble_raw_row(
            context,
            executor_capability=executor_capability,
            training=training,
            checkpoint=checkpoint,
            evaluation=evaluation,
            heldout_media=heldout_media,
            wandb_run_file=wandb_receipt,
            row_measurements=row_measurements,
        )
        output = _verify_and_publish_raw_row(context, row)
        return {
            "status": "measured",
            "row": _display(output),
            "row_sha256": file_sha256(output),
            "preflight": details,
        }
    except BaseException:
        request.output_path.unlink(missing_ok=True)
        if media_sink is not None:
            media_sink.abort()
        raise
    finally:
        if row_memory_sampler is not None:
            row_memory_sampler.stop()
        if session is not None:
            session.close()
        if dataset is not None:
            dataset.close()


def build_row_plan(request: RowRequest) -> dict[str, Any]:
    blockers, details = preflight_blockers(request)
    payload = {
        "schema_version": 1,
        "kind": "worldfoam-native4d-public-quality-row-plan-v1",
        "scene": request.scene,
        "seed": request.seed,
        "route": request.route,
        "output": _display(request.output_path),
        "runtime_ready": not blockers,
        "allocation_started": False,
        "blockers": blockers,
        "details": details,
    }
    return {**payload, "plan_sha256": _sha256(payload)}


__all__ = [
    "CalibratedPublicQualityDataset",
    "FullPixelWorkPlan",
    "HeldoutEvaluationReceipt",
    "MAXIMUM_PIXELS_PER_CHUNK",
    "PixelChunkPayload",
    "PixelChunkRequest",
    "ProductionRouteExecutor",
    "ProductionRouteSession",
    "PublicQualityDataset",
    "ROUTE_EXECUTOR_MODULES",
    "RowContext",
    "RowRequest",
    "StreamingSideBySideMp4Sink",
    "WORLD_FOAM_TRAINING_INPUT_KEYS",
    "WorldFoamTrainingInputsDataset",
    "assemble_raw_row",
    "build_full_pixel_work_plan",
    "build_row_plan",
    "evaluate_final_checkpoint",
    "execute_row_lifecycle",
    "load_dataset_capability",
    "load_public_quality_dataset",
    "preflight_blockers",
    "resolve_row_request",
    "run_training_lifecycle",
    "validate_training_receipt",
    "validate_worldfoam_training_inputs",
]
