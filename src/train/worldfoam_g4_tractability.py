"""Allocation-free tractability audit for the frozen WorldFoam G4 schedule.

Peak-memory boundedness and schedule tractability are separate gates.  The
current exact active-track compiler performs a cold compile for every unique
``(view, pixel)`` after each geometry update, validates the complete camera
path, admits the full site set, and retains no neighboring-track template.
This module derives exact deterministic scheduler counts plus separately
labelled compiler-bound totals, so row preflight and executor capability fail
closed on the same evidence without presenting a worst-case materialization
bound as an observed runtime count or a maximum-live-memory count.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


TRACTABILITY_SCHEMA_VERSION = 1
TRACTABILITY_KIND = "worldfoam-g4-full-schedule-tractability-v1"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class WorldFoamG4TractabilityAudit:
    optimizer_steps: int
    sampled_observation_count: int
    cold_track_compile_count: int
    cold_track_compile_upper_bound: int
    spatial_bundle_count: int
    spatial_bundle_upper_bound: int
    framewise_native_step_call_count: int
    complete_camera_record_validation_count: int
    total_admitted_site_reference_upper_bound: int
    total_compiled_chart_row_upper_bound: int
    total_shared_native_block_upper_bound: int
    total_framewise_native_block_upper_bound: int
    compiled_program_cache_retained: bool
    certified_spatial_pruning_or_cross_pixel_reuse: bool
    full_schedule_tractability_attested: bool
    generation_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRACTABILITY_SCHEMA_VERSION,
            "kind": TRACTABILITY_KIND,
            "optimizer_steps": self.optimizer_steps,
            "sampled_observation_count": self.sampled_observation_count,
            "cold_track_compile_count": self.cold_track_compile_count,
            "cold_track_compile_upper_bound": (
                self.cold_track_compile_upper_bound
            ),
            "spatial_bundle_count": self.spatial_bundle_count,
            "spatial_bundle_upper_bound": self.spatial_bundle_upper_bound,
            "framewise_native_step_call_count": (
                self.framewise_native_step_call_count
            ),
            "complete_camera_record_validation_count": (
                self.complete_camera_record_validation_count
            ),
            "total_admitted_site_reference_upper_bound": (
                self.total_admitted_site_reference_upper_bound
            ),
            "total_compiled_chart_row_upper_bound": (
                self.total_compiled_chart_row_upper_bound
            ),
            "total_shared_native_block_upper_bound": (
                self.total_shared_native_block_upper_bound
            ),
            "total_framewise_native_block_upper_bound": (
                self.total_framewise_native_block_upper_bound
            ),
            "compiled_program_cache_retained": (
                self.compiled_program_cache_retained
            ),
            "certified_spatial_pruning_or_cross_pixel_reuse": (
                self.certified_spatial_pruning_or_cross_pixel_reuse
            ),
            "full_schedule_tractability_attested": (
                self.full_schedule_tractability_attested
            ),
            "generation_digest": self.generation_digest,
        }

    @property
    def blocker(self) -> str | None:
        if self.full_schedule_tractability_attested:
            return None
        return (
            "worldfoam_full_schedule_tractability_unattested:"
            f"cold_track_compiles={self.cold_track_compile_count};"
            "cold_track_compile_upper_bound="
            f"{self.cold_track_compile_upper_bound};"
            f"spatial_bundles={self.spatial_bundle_count};"
            "framewise_native_step_calls="
            f"{self.framewise_native_step_call_count};"
            "certified_spatial_pruning_or_cross_pixel_reuse=false"
        )


def audit_worldfoam_g4_full_schedule(
    *,
    protocol: Any,
    work_plan: Any,
    compiler: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> WorldFoamG4TractabilityAudit:
    """Derive exact schedule counts without importing Torch or reading data."""

    tracks_per_bundle = int(runtime["maximum_tracks_per_bundle"])
    rows_per_native_block = int(runtime["maximum_rows_per_native_block"])
    maximum_sites_per_track = int(compiler["maximum_sites_per_track_compile"])
    maximum_charts_per_track = int(compiler["maximum_charts_per_track"])
    if min(
        tracks_per_bundle,
        rows_per_native_block,
        maximum_sites_per_track,
        maximum_charts_per_track,
    ) < 1:
        raise ValueError("WorldFoam tractability bounds must be positive")

    cold_track_compiles = 0
    cold_track_compile_upper_bound = 0
    spatial_bundles = 0
    spatial_bundle_upper_bound = 0
    framewise_native_step_calls = 0
    total_shared_native_block_upper_bound = 0
    total_framewise_native_block_upper_bound = 0
    for work in work_plan.steps:
        image_pixels = int(work.stage.image_size.pixels)
        samples = tuple(work.batch.samples)
        selected_views = {int(sample.view_index) for sample in samples}
        maximum_selected_views = min(
            len(protocol.dataset.train_cameras),
            len(samples),
        )
        chunks_per_view = math.ceil(image_pixels / tracks_per_bundle)
        cold_track_compiles += len(selected_views) * image_pixels
        cold_track_compile_upper_bound += maximum_selected_views * image_pixels
        spatial_bundles += len(selected_views) * chunks_per_view
        spatial_bundle_upper_bound += maximum_selected_views * chunks_per_view
        framewise_native_step_calls += len(samples) * chunks_per_view

        full_chunks, remainder = divmod(image_pixels, tracks_per_bundle)
        maximum_blocks_per_view = full_chunks * math.ceil(
            tracks_per_bundle * maximum_charts_per_track
            / rows_per_native_block
        )
        if remainder:
            maximum_blocks_per_view += math.ceil(
                remainder * maximum_charts_per_track / rows_per_native_block
            )
        total_shared_native_block_upper_bound += (
            len(selected_views) * maximum_blocks_per_view
        )
        total_framewise_native_block_upper_bound += (
            len(samples) * maximum_blocks_per_view
        )

    sampled_observations = int(work_plan.target_pixels)
    if (
        cold_track_compiles < 1
        or cold_track_compiles > cold_track_compile_upper_bound
        or spatial_bundles < 1
        or spatial_bundles > spatial_bundle_upper_bound
        or framewise_native_step_calls < 1
        or sampled_observations < 1
    ):
        raise ArithmeticError("WorldFoam tractability audit found invalid coverage")

    values = {
        "optimizer_steps": len(work_plan.steps),
        "sampled_observation_count": sampled_observations,
        "cold_track_compile_count": cold_track_compiles,
        "cold_track_compile_upper_bound": cold_track_compile_upper_bound,
        "spatial_bundle_count": spatial_bundles,
        "spatial_bundle_upper_bound": spatial_bundle_upper_bound,
        "framewise_native_step_call_count": framewise_native_step_calls,
        "complete_camera_record_validation_count": (
            cold_track_compiles * int(protocol.dataset.frame_count)
        ),
        "total_admitted_site_reference_upper_bound": (
            cold_track_compiles * maximum_sites_per_track
        ),
        "total_compiled_chart_row_upper_bound": (
            cold_track_compiles * maximum_charts_per_track
        ),
        "total_shared_native_block_upper_bound": (
            total_shared_native_block_upper_bound
        ),
        "total_framewise_native_block_upper_bound": (
            total_framewise_native_block_upper_bound
        ),
        # These are source facts of the selected v1 active-track factory.  A
        # v2 spatial compiler must replace this audit rather than flip config.
        "compiled_program_cache_retained": False,
        "certified_spatial_pruning_or_cross_pixel_reuse": False,
        "full_schedule_tractability_attested": False,
    }
    return WorldFoamG4TractabilityAudit(
        **values,
        generation_digest=_canonical_sha256(
            {
                "schema_version": TRACTABILITY_SCHEMA_VERSION,
                "kind": TRACTABILITY_KIND,
                **values,
            }
        ),
    )


__all__ = (
    "TRACTABILITY_KIND",
    "TRACTABILITY_SCHEMA_VERSION",
    "WorldFoamG4TractabilityAudit",
    "audit_worldfoam_g4_full_schedule",
)
