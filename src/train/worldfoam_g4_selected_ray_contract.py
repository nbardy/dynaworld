"""Deterministic selected-ray workload contract for WorldFoam G4-v2.

G4-v1 remains the frozen all-pixel experiment.  This module defines a separate
matched training schedule for all four routes: every sampled spacetime image
uses the same route-independent sensor-pixel set, while the final heldout
evaluation remains all-pixel and all-300-frame.  The module is allocation-light
and imports neither Torch nor a native extension.

The receipt proves scheduler and compiler-work counts.  It does *not* prove
native runtime, memory fit, quality, or that route-specific rasterized work is
equal.  Those remain measured row fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config_utils import load_config_file
from paper_training_protocol import (
    PaperSampleScheduleDigest,
    SpacetimeEpochSampler,
    paper_stage_for_step,
    resolve_paper_training_protocol,
)
from paper_training_types import PaperTrainingProtocol


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT
    / "src"
    / "train_configs"
    / "paper_protocols"
    / "worldfoam_native4d_g4_public_quality_v2_selected_rays.jsonc"
)
CONTRACT_SCHEMA_VERSION = 2
CONTRACT_KIND = "worldfoam-native4d-public-quality-selected-ray-ablation-v2"
WORKLOAD_SCHEMA_VERSION = 1
WORKLOAD_KIND = "worldfoam-g4-v2-selected-ray-workload-receipt-v1"
SELECTION_KIND = "step_shared_bijective_sensor_pixel_blocks_v1"

REQUIRED_ROUTES = (
    "worldfoam_native4d",
    "worldfoam_framewise_replay",
    "world_tubes",
    "dynamic_3dgs",
)
REQUIRED_SCENES = (
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
)
REQUIRED_SEEDS = (17, 29, 43)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "artifact_kind",
    "output_root",
    "base_g4_v1",
    "matrix",
    "training_sampling",
    "training_loss",
    "fairness",
    "evaluation",
    "expected_workload",
    "tractability_limits",
    "execution",
}
_TRAINING_LOSS_KEYS = {"identifier", "formula", "normalization"}
_BASE_KEYS = {"path", "expected_sha256"}
_MATRIX_KEYS = {
    "scene_order",
    "seed_order",
    "route_order",
    "expected_row_count",
    "fresh_process_per_row",
}
_SAMPLING_KEYS = {
    "kind",
    "pixels_per_spacetime_sample",
    "maximum_selected_pixels_per_chunk",
    "same_pixel_set_across_batch_samples",
    "without_replacement_within_sample",
    "permutation_multiplier",
    "rotation_derivation",
    "step_advance",
    "pixel_order_for_consumers",
    "route_specific_pixel_sampling_permitted",
}
_FAIRNESS_KEYS = {
    "identical_spacetime_schedule_within_scene_seed",
    "identical_selected_pixel_schedule_within_scene_seed",
    "identical_target_and_loss_pixel_budget_all_routes",
    "identical_optimizer_steps_all_routes",
    "identical_final_heldout_metric_and_coverage_contract_all_routes",
    "route_specific_training_rasterized_work_must_be_reported",
    "training_rasterized_work_is_claimed_equal",
    "reduced_pixels_only_for_worldfoam_permitted",
    "all_pixel_g4_v1_mutated",
}
_EVALUATION_KEYS = {
    "full_temporal_heldout_evaluation",
    "full_pixel_heldout_evaluation",
    "heldout_frame_count",
    "heldout_image_size",
    "heldout_camera_count_per_scene",
    "heldout_target_pixels_per_row",
    "selected_ray_training_metrics_are_not_heldout_metrics",
}
_EXPECTED_KEYS = {
    "optimizer_steps",
    "spacetime_samples_per_step",
    "selected_target_pixels_per_step",
    "selected_target_pixels_per_row",
    "selected_loss_scalars_per_row",
    "unique_sensor_pixel_coverage_per_row",
    "minimum_sensor_pixel_visit_count",
    "maximum_sensor_pixel_visit_count",
    "all_pixel_cold_compile_reduction_factor",
}
_LIMIT_KEYS = {
    "maximum_cold_track_compile_count",
    "maximum_complete_camera_record_validation_count",
    "maximum_admitted_site_reference_upper_bound",
    "maximum_spatial_bundle_count",
    "maximum_framewise_native_step_call_count",
    "maximum_heldout_spatial_major_render_call_count",
    "maximum_heldout_native_bundle_count",
    "maximum_heldout_complete_camera_record_validation_count",
    "maximum_heldout_admitted_site_reference_upper_bound",
    "minimum_all_pixel_cold_compile_reduction_factor",
}
_EXECUTION_KEYS = {
    "row_worker",
    "source_capability",
    "pilot_receipt",
    "pilot_scene",
    "pilot_seed",
    "pilot_required_routes",
    "maximum_projected_worldfoam_row_hours",
    "spatial_major_full_temporal_heldout_required",
    "process_group_watchdog_required",
    "maximum_worker_process_group_rss_bytes",
    "worker_watchdog_poll_interval_seconds",
    "worker_timeout_seconds",
    "pre_matrix_host_resource_guard_required",
    "pre_matrix_minimum_free_disk_bytes",
    "pre_matrix_minimum_available_memory_bytes",
    "pre_matrix_maximum_swap_used_bytes",
    "pre_matrix_maximum_load_average",
    "maximum_mps_working_set_bytes_per_worker",
    "minimum_free_disk_bytes_before_worldfoam_row",
    "real_native_only",
    "proxy_or_smoke_evidence_permitted",
    "abort_before_first_row_on_any_blocker",
    "local_mps_execution_requires_explicit_acknowledgement",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{name} keys changed: missing={missing}, extra={extra}")


def _repo_path(value: Any, *, name: str) -> Path:
    path = Path(str(value))
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{name} left the repository") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is missing: {resolved}")
    return resolved


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _selection_rotation(*, sample_id: str, seed: int, image_pixels: int) -> int:
    digest = hashlib.sha256(
        _canonical_bytes(
            {
                "kind": SELECTION_KIND,
                "sample_id": sample_id,
                "seed": int(seed),
            }
        )
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % image_pixels


def selected_pixel_ids(
    *,
    image_pixels: int,
    pixels_per_sample: int,
    permutation_multiplier: int,
    rotation: int,
    step: int,
) -> tuple[int, ...]:
    """Return one sorted, unique sensor-pixel set without an image allocation."""

    image_pixels = _positive_int(image_pixels, name="image_pixels")
    pixels_per_sample = _positive_int(
        pixels_per_sample, name="pixels_per_sample"
    )
    permutation_multiplier = _positive_int(
        permutation_multiplier, name="permutation_multiplier"
    )
    if pixels_per_sample > image_pixels:
        raise ValueError("pixels_per_sample exceeds the sensor grid")
    if math.gcd(permutation_multiplier, image_pixels) != 1:
        raise ValueError("pixel permutation multiplier is not bijective")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    start = (int(rotation) + step * pixels_per_sample) % image_pixels
    result = tuple(
        sorted(
            permutation_multiplier * ((start + offset) % image_pixels)
            % image_pixels
            for offset in range(pixels_per_sample)
        )
    )
    if len(result) != pixels_per_sample or len(set(result)) != len(result):
        raise ArithmeticError("selected pixel block is not unique")
    return result


@dataclass(frozen=True)
class WorldFoamG4SelectedRayWorkloadReceipt:
    scene: str
    sample_id: str
    seed: int
    protocol_path: str
    protocol_sha256: str
    v2_config_sha256: str
    base_g4_v1_sha256: str
    training_loss_identifier: str
    training_loss_contract_sha256: str
    optimizer_steps: int
    spacetime_samples_per_step: int
    sampled_spacetime_image_count: int
    image_height: int
    image_width: int
    selected_pixels_per_spacetime_sample: int
    selected_target_pixels_per_step: int
    selected_target_pixels: int
    selected_loss_scalar_count: int
    selected_pixel_chunk_count: int
    unique_sensor_pixel_coverage: int
    minimum_sensor_pixel_visit_count: int
    maximum_sensor_pixel_visit_count: int
    heldout_frame_count: int
    heldout_target_pixels: int
    heldout_cross_time_track_block_size: int
    heldout_spatial_major_render_call_count: int
    heldout_native_bundle_count: int
    heldout_cold_track_compile_count: int
    heldout_complete_camera_record_validation_count: int
    heldout_admitted_site_reference_upper_bound: int
    cold_track_compile_count: int
    all_pixel_v1_cold_track_compile_count: int
    all_pixel_cold_compile_reduction_factor: float
    complete_camera_record_validation_count: int
    admitted_site_reference_upper_bound: int
    compiled_chart_row_upper_bound: int
    spatial_bundle_count: int
    framewise_native_step_call_count: int
    shared_native_block_upper_bound: int
    framewise_native_block_upper_bound: int
    spacetime_schedule_sha256: str
    selected_pixel_schedule_sha256: str
    selected_track_schedule_sha256: str
    sample_schedule_sha256: str
    route_schedule_sha256: str
    tractability_preflight_passed: bool
    tractability_failures: tuple[str, ...]
    generation_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORKLOAD_SCHEMA_VERSION,
            "kind": WORKLOAD_KIND,
            **{
                key: value
                for key, value in self.__dict__.items()
                if key not in {"tractability_failures"}
            },
            "tractability_failures": list(self.tractability_failures),
            "selected_pixel_set_shared_across_step_batch": True,
            "selected_pixel_sampling_without_replacement": True,
            "identical_target_schedule_required_routes": list(REQUIRED_ROUTES),
            "route_specific_training_rasterized_work_claimed_equal": False,
            "route_specific_training_rasterized_work_receipt_required": True,
            "full_pixel_full_temporal_heldout_evaluation": True,
            "runtime_or_memory_measured": False,
            "public_quality_evidence": False,
        }


def _receipt_digest(values: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": WORKLOAD_SCHEMA_VERSION,
            "kind": WORKLOAD_KIND,
            **values,
        }
    )


def _validate_base_contract(base: Mapping[str, Any]) -> None:
    if (
        base.get("schema_version") != 1
        or base.get("name") != "worldfoam_native4d_g4_public_quality_v1"
        or base.get("artifact_kind")
        != "worldfoam-native4d-public-quality-ablation-v1"
        or base.get("device") != "mps"
        or tuple(base.get("seeds", ())) != REQUIRED_SEEDS
    ):
        raise ValueError("bound G4-v1 identity changed")
    scenes = base.get("scenes")
    routes = base.get("routes")
    if not isinstance(scenes, list) or tuple(
        str(scene.get("scene")) for scene in scenes if isinstance(scene, Mapping)
    ) != REQUIRED_SCENES:
        raise ValueError("bound G4-v1 scene order changed")
    if not isinstance(routes, list) or tuple(
        str(route.get("route")) for route in routes if isinstance(route, Mapping)
    ) != REQUIRED_ROUTES:
        raise ValueError("bound G4-v1 route order changed")
    public = _mapping(base.get("public_protocol"), name="base public protocol")
    if (
        public.get("dataset_frame_count") != 300
        or public.get("image_size") != [384, 512]
        or public.get("optimizer_steps") != 300
        or public.get("frames_per_step") != 4
        or public.get("primitive_count") != 1024
        or public.get("require_final_checkpoint_heldout_evaluation") is not True
        or public.get("require_full_temporal_heldout_evaluation") is not True
        or public.get("require_identical_evaluator_within_scene_seed") is not True
        or public.get("require_identical_sample_schedule_within_scene_seed")
        is not True
    ):
        raise ValueError("bound G4-v1 public protocol changed")


def load_selected_ray_contract(
    path: Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Load and fail-closed validate the v2 overlay and immutable G4-v1 base."""

    config_path = Path(path).resolve()
    config = load_config_file(config_path)
    _exact_keys(config, _TOP_LEVEL_KEYS, name="G4-v2 contract")
    if (
        config.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or config.get("artifact_kind") != CONTRACT_KIND
        or config.get("name")
        != "worldfoam_native4d_g4_public_quality_v2_selected_rays"
        or not str(config.get("output_root", "")).strip()
    ):
        raise ValueError("G4-v2 contract identity changed")

    base_spec = _mapping(config["base_g4_v1"], name="base_g4_v1")
    matrix = _mapping(config["matrix"], name="matrix")
    sampling = _mapping(config["training_sampling"], name="training_sampling")
    training_loss = _mapping(config["training_loss"], name="training_loss")
    fairness = _mapping(config["fairness"], name="fairness")
    evaluation = _mapping(config["evaluation"], name="evaluation")
    expected = _mapping(config["expected_workload"], name="expected_workload")
    limits = _mapping(config["tractability_limits"], name="tractability_limits")
    execution = _mapping(config["execution"], name="execution")
    for value, keys, name in (
        (base_spec, _BASE_KEYS, "base_g4_v1"),
        (matrix, _MATRIX_KEYS, "matrix"),
        (sampling, _SAMPLING_KEYS, "training_sampling"),
        (training_loss, _TRAINING_LOSS_KEYS, "training_loss"),
        (fairness, _FAIRNESS_KEYS, "fairness"),
        (evaluation, _EVALUATION_KEYS, "evaluation"),
        (expected, _EXPECTED_KEYS, "expected_workload"),
        (limits, _LIMIT_KEYS, "tractability_limits"),
        (execution, _EXECUTION_KEYS, "execution"),
    ):
        _exact_keys(value, keys, name=name)

    base_path = _repo_path(base_spec["path"], name="base G4-v1 config")
    if file_sha256(base_path) != base_spec["expected_sha256"]:
        raise ValueError("bound G4-v1 file digest changed")
    base = load_config_file(base_path)
    _validate_base_contract(base)
    if (
        tuple(matrix["scene_order"]) != REQUIRED_SCENES
        or tuple(matrix["seed_order"]) != REQUIRED_SEEDS
        or tuple(matrix["route_order"]) != REQUIRED_ROUTES
        or matrix["expected_row_count"] != 36
        or matrix["fresh_process_per_row"] is not True
    ):
        raise ValueError("G4-v2 matrix changed")

    required_true = (
        "identical_spacetime_schedule_within_scene_seed",
        "identical_selected_pixel_schedule_within_scene_seed",
        "identical_target_and_loss_pixel_budget_all_routes",
        "identical_optimizer_steps_all_routes",
        "identical_final_heldout_metric_and_coverage_contract_all_routes",
        "route_specific_training_rasterized_work_must_be_reported",
    )
    if any(fairness[key] is not True for key in required_true) or any(
        fairness[key] is not False
        for key in (
            "training_rasterized_work_is_claimed_equal",
            "reduced_pixels_only_for_worldfoam_permitted",
            "all_pixel_g4_v1_mutated",
        )
    ):
        raise ValueError("G4-v2 fairness contract changed")
    if (
        sampling["kind"] != SELECTION_KIND
        or sampling["same_pixel_set_across_batch_samples"] is not True
        or sampling["without_replacement_within_sample"] is not True
        or sampling["route_specific_pixel_sampling_permitted"] is not False
        or sampling["pixel_order_for_consumers"] != "ascending_pixel_id"
    ):
        raise ValueError("G4-v2 pixel selection semantics changed")
    if dict(training_loss) != {
        "identifier": "rgb_mse_mean_v1",
        "formula": "mean((prediction-target)^2)",
        "normalization": "mean_over_selected_rgb_scalars",
    }:
        raise ValueError("G4-v2 requires identical RGB-MSE on all four routes")
    pixels = 384 * 512
    if math.gcd(int(sampling["permutation_multiplier"]), pixels) != 1:
        raise ValueError("G4-v2 pixel permutation is not bijective")
    if (
        evaluation["full_temporal_heldout_evaluation"] is not True
        or evaluation["full_pixel_heldout_evaluation"] is not True
        or evaluation["heldout_frame_count"] != 300
        or evaluation["heldout_image_size"] != [384, 512]
        or evaluation["heldout_camera_count_per_scene"] != 1
        or evaluation["heldout_target_pixels_per_row"] != 300 * pixels
        or evaluation["selected_ray_training_metrics_are_not_heldout_metrics"]
        is not True
    ):
        raise ValueError("G4-v2 heldout evaluator contract changed")
    if (
        execution["real_native_only"] is not True
        or execution["proxy_or_smoke_evidence_permitted"] is not False
        or execution["abort_before_first_row_on_any_blocker"] is not True
        or execution["local_mps_execution_requires_explicit_acknowledgement"]
        is not True
        or execution["pilot_scene"] != "coffee_martini"
        or execution["pilot_seed"] != 17
        or execution["pilot_required_routes"]
        != ["worldfoam_native4d", "worldfoam_framewise_replay"]
        or execution["spatial_major_full_temporal_heldout_required"] is not True
        or execution["process_group_watchdog_required"] is not True
        or execution["maximum_worker_process_group_rss_bytes"] != 4 * 1024**3
        or execution["worker_watchdog_poll_interval_seconds"] != 0.25
        or execution["worker_timeout_seconds"] != 12 * 60 * 60
        or execution["pre_matrix_host_resource_guard_required"] is not True
        or execution["pre_matrix_minimum_free_disk_bytes"] != 8 * 1024**3
        or execution["pre_matrix_minimum_available_memory_bytes"] != 8 * 1024**3
        or execution["pre_matrix_maximum_swap_used_bytes"] != 2 * 1024**3
        or execution["pre_matrix_maximum_load_average"] != 8.0
        or execution["maximum_mps_working_set_bytes_per_worker"] != 2 * 1024**3
        or execution["minimum_free_disk_bytes_before_worldfoam_row"]
        != 2 * 1024**3
        or not isinstance(execution["maximum_projected_worldfoam_row_hours"], (int, float))
        or isinstance(execution["maximum_projected_worldfoam_row_hours"], bool)
        or float(execution["maximum_projected_worldfoam_row_hours"]) <= 0.0
    ):
        raise ValueError("G4-v2 execution gate changed")
    for key, value in {**expected, **limits}.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"G4-v2 numeric contract {key} must be positive")
    return config, base, base_path


def _scene_spec(base: Mapping[str, Any], scene: str) -> Mapping[str, Any]:
    matches = [
        value
        for value in base["scenes"]
        if isinstance(value, Mapping) and value.get("scene") == scene
    ]
    if len(matches) != 1:
        raise ValueError(f"base G4-v1 has no unique scene {scene!r}")
    return matches[0]


def build_selected_ray_workload_receipt(
    *,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    config_path: Path,
    base_path: Path,
    scene: str,
    seed: int,
) -> WorldFoamG4SelectedRayWorkloadReceipt:
    """Build one exact scene/seed workload receipt without loading pixels."""

    if scene not in REQUIRED_SCENES or int(seed) not in REQUIRED_SEEDS:
        raise ValueError("G4-v2 workload key left the frozen matrix")
    scene_spec = _scene_spec(base, scene)
    protocol_path = _repo_path(scene_spec["protocol"], name="scene protocol")
    protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
    if (
        protocol.steps != 300
        or protocol.final_stage.image_size.as_list() != [384, 512]
        or len(protocol.stages) != 1
        or protocol.stages[0].frames_per_step != 4
        or protocol.stages[0].primitive_count != 1024
        or len(protocol.dataset.heldout_cameras) != 1
    ):
        raise ValueError("G4-v2 requires the fixed 300-step 512-wide protocol")

    sampling = config["training_sampling"]
    expected = config["expected_workload"]
    limits = config["tractability_limits"]
    runtime = base["worldfoam_runtime"]
    compiler = base["compiler"]
    pixels_per_sample = int(sampling["pixels_per_spacetime_sample"])
    maximum_chunk = int(sampling["maximum_selected_pixels_per_chunk"])
    multiplier = int(sampling["permutation_multiplier"])
    image_pixels = protocol.final_stage.image_size.pixels
    rotation = _selection_rotation(
        sample_id=protocol.dataset.sample_id,
        seed=int(seed),
        image_pixels=image_pixels,
    )
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
    selected_pixel_digest = hashlib.sha256()
    selected_track_digest = hashlib.sha256()
    sampled_images = 0
    selected_targets = 0
    selected_chunks = 0
    cold_tracks = 0
    all_pixel_cold_tracks = 0
    spatial_bundles = 0
    framewise_calls = 0
    shared_blocks = 0
    framewise_blocks = 0
    tracks_per_bundle = int(runtime["maximum_tracks_per_bundle"])
    rows_per_block = int(runtime["maximum_rows_per_native_block"])
    max_charts = int(compiler["maximum_charts_per_track"])
    chunks_per_view = math.ceil(pixels_per_sample / tracks_per_bundle)
    # A host prediction request can use the full spatial-track bound.  The
    # provider streams that request through smaller native bundles constrained
    # by the observation cap.  Keep host calls and native bundles separate so
    # Python/MPS dispatch overhead cannot be hidden in the compiler count.
    heldout_block_size = tracks_per_bundle
    heldout_tracks_per_native_bundle = min(
        tracks_per_bundle,
        int(runtime["maximum_observations_per_bundle"])
        // protocol.dataset.frame_count,
    )
    if heldout_tracks_per_native_bundle < 1:
        raise ValueError("WorldFoam heldout bundle cannot admit one full-time track")

    for step in range(protocol.steps):
        stage = paper_stage_for_step(protocol.stages, step)
        batch = sampler.next_batch(stage.frames_per_step)
        schedule.record(step=step, stage=stage, batch=batch)
        pixel_ids = selected_pixel_ids(
            image_pixels=stage.image_size.pixels,
            pixels_per_sample=pixels_per_sample,
            permutation_multiplier=multiplier,
            rotation=rotation,
            step=step,
        )
        pixel_set_sha256 = canonical_sha256(pixel_ids)
        selected_pixel_digest.update(
            _canonical_bytes(
                {
                    "step": step,
                    "pixel_count": pixels_per_sample,
                    "pixel_set_sha256": pixel_set_sha256,
                }
            )
        )
        selected_pixel_digest.update(b"\n")
        selected_views = tuple(sorted({sample.view_index for sample in batch.samples}))
        for view_index in selected_views:
            selected_track_digest.update(
                _canonical_bytes(
                    {
                        "step": step,
                        "view_index": view_index,
                        "pixel_set_sha256": pixel_set_sha256,
                    }
                )
            )
            selected_track_digest.update(b"\n")
        for sample_slot, sample in enumerate(batch.samples):
            selected_pixel_digest.update(
                _canonical_bytes(
                    {
                        "step": step,
                        "sample_slot": sample_slot,
                        "view_index": sample.view_index,
                        "frame_index": sample.frame_index,
                        "pixel_set_sha256": pixel_set_sha256,
                    }
                )
            )
            selected_pixel_digest.update(b"\n")
        sampled_images += len(batch.samples)
        selected_targets += len(batch.samples) * pixels_per_sample
        selected_chunks += len(batch.samples) * math.ceil(
            pixels_per_sample / maximum_chunk
        )
        cold_tracks += len(selected_views) * pixels_per_sample
        all_pixel_cold_tracks += len(selected_views) * image_pixels
        spatial_bundles += len(selected_views) * chunks_per_view
        framewise_calls += len(batch.samples) * chunks_per_view

        full_chunks, remainder = divmod(pixels_per_sample, tracks_per_bundle)
        blocks_per_view = full_chunks * math.ceil(
            tracks_per_bundle * max_charts / rows_per_block
        )
        if remainder:
            blocks_per_view += math.ceil(remainder * max_charts / rows_per_block)
        shared_blocks += len(selected_views) * blocks_per_view
        framewise_blocks += len(batch.samples) * blocks_per_view

    schedule_receipt = schedule.snapshot()
    sensor_selection_count = protocol.steps * pixels_per_sample
    minimum_visits, remainder_visits = divmod(sensor_selection_count, image_pixels)
    maximum_visits = minimum_visits + int(remainder_visits > 0)
    unique_coverage = min(sensor_selection_count, image_pixels)
    reduction = all_pixel_cold_tracks / cold_tracks
    validation_count = cold_tracks * protocol.dataset.frame_count
    site_references = cold_tracks * int(compiler["maximum_sites_per_track_compile"])
    chart_rows = cold_tracks * max_charts
    heldout_cold_tracks = (
        len(protocol.dataset.heldout_cameras) * image_pixels
    )
    heldout_render_calls = (
        len(protocol.dataset.heldout_cameras)
        * math.ceil(image_pixels / heldout_block_size)
    )
    heldout_native_bundles = sum(
        math.ceil(
            min(heldout_block_size, image_pixels - pixel_start)
            / heldout_tracks_per_native_bundle
        )
        for pixel_start in range(0, image_pixels, heldout_block_size)
    ) * len(protocol.dataset.heldout_cameras)
    heldout_validations = heldout_cold_tracks * protocol.dataset.frame_count
    heldout_site_references = (
        heldout_cold_tracks * int(compiler["maximum_sites_per_track_compile"])
    )
    failures: list[str] = []
    comparisons = (
        (
            cold_tracks,
            int(limits["maximum_cold_track_compile_count"]),
            "cold_track_compile_count",
        ),
        (
            validation_count,
            int(limits["maximum_complete_camera_record_validation_count"]),
            "complete_camera_record_validation_count",
        ),
        (
            site_references,
            int(limits["maximum_admitted_site_reference_upper_bound"]),
            "admitted_site_reference_upper_bound",
        ),
        (
            spatial_bundles,
            int(limits["maximum_spatial_bundle_count"]),
            "spatial_bundle_count",
        ),
        (
            framewise_calls,
            int(limits["maximum_framewise_native_step_call_count"]),
            "framewise_native_step_call_count",
        ),
        (
            heldout_render_calls,
            int(limits["maximum_heldout_spatial_major_render_call_count"]),
            "heldout_spatial_major_render_call_count",
        ),
        (
            heldout_native_bundles,
            int(limits["maximum_heldout_native_bundle_count"]),
            "heldout_native_bundle_count",
        ),
        (
            heldout_validations,
            int(limits["maximum_heldout_complete_camera_record_validation_count"]),
            "heldout_complete_camera_record_validation_count",
        ),
        (
            heldout_site_references,
            int(limits["maximum_heldout_admitted_site_reference_upper_bound"]),
            "heldout_admitted_site_reference_upper_bound",
        ),
    )
    failures.extend(name for value, limit, name in comparisons if value > limit)
    if reduction < float(limits["minimum_all_pixel_cold_compile_reduction_factor"]):
        failures.append("all_pixel_cold_compile_reduction_factor")

    expected_values = {
        "optimizer_steps": protocol.steps,
        "spacetime_samples_per_step": protocol.stages[0].frames_per_step,
        "selected_target_pixels_per_step": (
            protocol.stages[0].frames_per_step * pixels_per_sample
        ),
        "selected_target_pixels_per_row": selected_targets,
        "selected_loss_scalars_per_row": selected_targets * 3,
        "unique_sensor_pixel_coverage_per_row": unique_coverage,
        "minimum_sensor_pixel_visit_count": minimum_visits,
        "maximum_sensor_pixel_visit_count": maximum_visits,
        "all_pixel_cold_compile_reduction_factor": reduction,
    }
    for key, value in expected_values.items():
        configured = expected[key]
        if isinstance(value, float):
            if not math.isclose(float(configured), value, rel_tol=0.0, abs_tol=0.0):
                raise ValueError(f"G4-v2 expected workload changed: {key}")
        elif configured != value:
            raise ValueError(f"G4-v2 expected workload changed: {key}")

    combined_schedule_sha256 = canonical_sha256(
        {
            "schema_version": WORKLOAD_SCHEMA_VERSION,
            "spacetime_schedule_sha256": schedule_receipt["sha256"],
            "selected_pixel_schedule_sha256": selected_pixel_digest.hexdigest(),
            "selected_track_schedule_sha256": selected_track_digest.hexdigest(),
            "target_pixels": selected_targets,
            "selection_kind": SELECTION_KIND,
        }
    )
    route_schedule_sha256 = canonical_sha256(
        {
            "sample_schedule_sha256": combined_schedule_sha256,
            "required_routes": REQUIRED_ROUTES,
            "identical_target_and_loss_pixels": True,
            "training_loss_contract_sha256": canonical_sha256(
                config["training_loss"]
            ),
            "training_rasterized_work_claimed_equal": False,
        }
    )
    values = {
        "scene": scene,
        "sample_id": protocol.dataset.sample_id,
        "seed": int(seed),
        "protocol_path": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "v2_config_sha256": file_sha256(config_path),
        "base_g4_v1_sha256": file_sha256(base_path),
        "training_loss_identifier": str(config["training_loss"]["identifier"]),
        "training_loss_contract_sha256": canonical_sha256(
            config["training_loss"]
        ),
        "optimizer_steps": protocol.steps,
        "spacetime_samples_per_step": protocol.stages[0].frames_per_step,
        "sampled_spacetime_image_count": sampled_images,
        "image_height": protocol.final_stage.image_size.height,
        "image_width": protocol.final_stage.image_size.width,
        "selected_pixels_per_spacetime_sample": pixels_per_sample,
        "selected_target_pixels_per_step": (
            protocol.stages[0].frames_per_step * pixels_per_sample
        ),
        "selected_target_pixels": selected_targets,
        "selected_loss_scalar_count": selected_targets * 3,
        "selected_pixel_chunk_count": selected_chunks,
        "unique_sensor_pixel_coverage": unique_coverage,
        "minimum_sensor_pixel_visit_count": minimum_visits,
        "maximum_sensor_pixel_visit_count": maximum_visits,
        "heldout_frame_count": int(config["evaluation"]["heldout_frame_count"]),
        "heldout_target_pixels": int(
            config["evaluation"]["heldout_target_pixels_per_row"]
        ),
        "heldout_cross_time_track_block_size": heldout_block_size,
        "heldout_spatial_major_render_call_count": heldout_render_calls,
        "heldout_native_bundle_count": heldout_native_bundles,
        "heldout_cold_track_compile_count": heldout_cold_tracks,
        "heldout_complete_camera_record_validation_count": heldout_validations,
        "heldout_admitted_site_reference_upper_bound": heldout_site_references,
        "cold_track_compile_count": cold_tracks,
        "all_pixel_v1_cold_track_compile_count": all_pixel_cold_tracks,
        "all_pixel_cold_compile_reduction_factor": reduction,
        "complete_camera_record_validation_count": validation_count,
        "admitted_site_reference_upper_bound": site_references,
        "compiled_chart_row_upper_bound": chart_rows,
        "spatial_bundle_count": spatial_bundles,
        "framewise_native_step_call_count": framewise_calls,
        "shared_native_block_upper_bound": shared_blocks,
        "framewise_native_block_upper_bound": framewise_blocks,
        "spacetime_schedule_sha256": str(schedule_receipt["sha256"]),
        "selected_pixel_schedule_sha256": selected_pixel_digest.hexdigest(),
        "selected_track_schedule_sha256": selected_track_digest.hexdigest(),
        "sample_schedule_sha256": combined_schedule_sha256,
        "route_schedule_sha256": route_schedule_sha256,
        "tractability_preflight_passed": not failures,
        "tractability_failures": tuple(failures),
    }
    receipt = WorldFoamG4SelectedRayWorkloadReceipt(
        **values,
        generation_digest=_receipt_digest(values),
    )
    if receipt.as_dict()["generation_digest"] != _receipt_digest(values):
        raise ArithmeticError("G4-v2 workload receipt digest changed")
    return receipt


def build_matrix_workload_receipts(
    path: Path = DEFAULT_CONFIG,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    dict[tuple[str, int], WorldFoamG4SelectedRayWorkloadReceipt],
]:
    config_path = Path(path).resolve()
    config, base, base_path = load_selected_ray_contract(config_path)
    receipts = {
        (scene, seed): build_selected_ray_workload_receipt(
            config=config,
            base=base,
            config_path=config_path,
            base_path=base_path,
            scene=scene,
            seed=seed,
        )
        for scene in REQUIRED_SCENES
        for seed in REQUIRED_SEEDS
    }
    return config, base, base_path, receipts


def iter_step_pixel_ids(
    *,
    config: Mapping[str, Any],
    protocol: PaperTrainingProtocol,
    seed: int,
) -> Iterator[tuple[int, tuple[int, ...]]]:
    """Public deterministic pixel-source seam for a future real row worker."""

    sampling = config["training_sampling"]
    image_pixels = protocol.final_stage.image_size.pixels
    rotation = _selection_rotation(
        sample_id=protocol.dataset.sample_id,
        seed=int(seed),
        image_pixels=image_pixels,
    )
    for step in range(protocol.steps):
        yield step, selected_pixel_ids(
            image_pixels=image_pixels,
            pixels_per_sample=int(sampling["pixels_per_spacetime_sample"]),
            permutation_multiplier=int(sampling["permutation_multiplier"]),
            rotation=rotation,
            step=step,
        )


__all__ = (
    "CONTRACT_KIND",
    "CONTRACT_SCHEMA_VERSION",
    "DEFAULT_CONFIG",
    "REQUIRED_ROUTES",
    "REQUIRED_SCENES",
    "REQUIRED_SEEDS",
    "SELECTION_KIND",
    "WORKLOAD_KIND",
    "WORKLOAD_SCHEMA_VERSION",
    "WorldFoamG4SelectedRayWorkloadReceipt",
    "build_matrix_workload_receipts",
    "build_selected_ray_workload_receipt",
    "canonical_sha256",
    "file_sha256",
    "iter_step_pixel_ids",
    "load_selected_ray_contract",
    "selected_pixel_ids",
)
