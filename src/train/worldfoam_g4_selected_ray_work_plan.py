"""Executable selected-ray work plan for the separate WorldFoam G4-v2 matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from paper_training_types import PaperTrainingProtocol
from worldfoam_g4_selected_ray_contract import (
    WorldFoamG4SelectedRayWorkloadReceipt,
    canonical_sha256,
    iter_step_pixel_ids,
)
from worldfoam_native4d_public_quality_row import (
    MAXIMUM_PIXELS_PER_CHUNK,
    FullPixelWorkPlan,
    PixelChunkRequest,
    StepWork,
)


WORK_PLAN_SCHEMA_VERSION = 2
WORK_PLAN_KIND = "worldfoam-g4-v2-selected-ray-work-plan-v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class SelectedRayWorkPlan:
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
    heldout_maximum_pixels_per_chunk: int
    selected_pixels_per_spacetime_sample: int
    selected_pixel_ids_by_step: tuple[tuple[int, ...], ...]
    training_loss_contract: Mapping[str, str]
    workload_receipt: WorldFoamG4SelectedRayWorkloadReceipt

    def selected_pixel_ids_for_step(self, step: int) -> tuple[int, ...]:
        if isinstance(step, bool) or not 0 <= int(step) < len(self.steps):
            raise IndexError("selected-ray step is outside the work plan")
        return self.selected_pixel_ids_by_step[int(step)]

    def iter_step_training_chunks(
        self,
        work: StepWork,
    ) -> Iterator[PixelChunkRequest]:
        if self.steps[work.step] is not work:
            raise ValueError("selected-ray work object is foreign or reordered")
        pixel_ids = self.selected_pixel_ids_for_step(work.step)
        for sample_slot, sample in enumerate(work.batch.samples):
            for selected_start in range(
                0,
                len(pixel_ids),
                self.maximum_pixels_per_chunk,
            ):
                selected = pixel_ids[
                    selected_start : selected_start + self.maximum_pixels_per_chunk
                ]
                yield PixelChunkRequest(
                    split="train",
                    step=work.step,
                    sample_slot=sample_slot,
                    camera_index=int(sample.view_index),
                    frame_index=int(sample.frame_index),
                    pixel_start=selected_start,
                    pixel_count=len(selected),
                    image_height=work.stage.image_size.height,
                    image_width=work.stage.image_size.width,
                    pixel_ids=selected,
                )

    def iter_training_chunks(self) -> Iterator[PixelChunkRequest]:
        for work in self.steps:
            yield from self.iter_step_training_chunks(work)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORK_PLAN_SCHEMA_VERSION,
            "kind": WORK_PLAN_KIND,
            "sampler_seed": self.sampler_seed,
            "optimizer_steps": len(self.steps),
            "sampled_image_count": self.sampled_image_count,
            "pixel_chunk_count": self.pixel_chunk_count,
            "target_pixels": self.target_pixels,
            "maximum_pixels_per_chunk": self.maximum_pixels_per_chunk,
            "heldout_maximum_pixels_per_chunk": (
                self.heldout_maximum_pixels_per_chunk
            ),
            "selected_pixels_per_spacetime_sample": (
                self.selected_pixels_per_spacetime_sample
            ),
            "spacetime_schedule_sha256": self.spacetime_schedule_sha256,
            "pixel_chunk_manifest_sha256": self.pixel_chunk_manifest_sha256,
            "sample_schedule_sha256": self.sample_schedule_sha256,
            "route_schedule_sha256": self.workload_receipt.route_schedule_sha256,
            "training_loss_contract": dict(self.training_loss_contract),
            "training_loss_contract_sha256": canonical_sha256(
                self.training_loss_contract
            ),
            "workload_receipt_generation_digest": (
                self.workload_receipt.generation_digest
            ),
        }


def build_selected_ray_work_plan(
    *,
    config: Mapping[str, Any],
    protocol: PaperTrainingProtocol,
    seed: int,
    full_pixel_plan: FullPixelWorkPlan,
    workload_receipt: WorldFoamG4SelectedRayWorkloadReceipt,
) -> SelectedRayWorkPlan:
    """Replace only the target-pixel stream; retain v1 spacetime steps exactly."""

    if (
        full_pixel_plan.protocol is not protocol
        or full_pixel_plan.seed != int(seed)
        or full_pixel_plan.spacetime_schedule_sha256
        != workload_receipt.spacetime_schedule_sha256
    ):
        raise ValueError("selected-ray plan differs from the bound spacetime schedule")
    emitted = tuple(
        iter_step_pixel_ids(config=config, protocol=protocol, seed=int(seed))
    )
    if tuple(step for step, _ids in emitted) != tuple(range(protocol.steps)):
        raise ArithmeticError("selected-pixel generator skipped or reordered a step")
    by_step = tuple(ids for _step, ids in emitted)
    maximum_chunk = int(config["training_sampling"]["maximum_selected_pixels_per_chunk"])
    pixel_manifest = hashlib.sha256()
    chunk_count = 0
    target_pixels = 0
    for work in full_pixel_plan.steps:
        ids = by_step[work.step]
        for sample_slot, sample in enumerate(work.batch.samples):
            for selected_start in range(0, len(ids), maximum_chunk):
                selected = ids[selected_start : selected_start + maximum_chunk]
                request = PixelChunkRequest(
                    split="train",
                    step=work.step,
                    sample_slot=sample_slot,
                    camera_index=int(sample.view_index),
                    frame_index=int(sample.frame_index),
                    pixel_start=selected_start,
                    pixel_count=len(selected),
                    image_height=work.stage.image_size.height,
                    image_width=work.stage.image_size.width,
                    pixel_ids=selected,
                )
                pixel_manifest.update(_canonical_bytes(request.as_dict()))
                pixel_manifest.update(b"\n")
                chunk_count += 1
                target_pixels += len(selected)
    if (
        target_pixels != workload_receipt.selected_target_pixels
        or chunk_count != workload_receipt.selected_pixel_chunk_count
        or len(full_pixel_plan.steps) != workload_receipt.optimizer_steps
        or full_pixel_plan.sampled_image_count
        != workload_receipt.sampled_spacetime_image_count
        or canonical_sha256(config["training_loss"])
        != workload_receipt.training_loss_contract_sha256
    ):
        raise ArithmeticError("selected-ray executable counts differ from the workload receipt")
    return SelectedRayWorkPlan(
        protocol=protocol,
        seed=int(seed),
        sampler_seed=full_pixel_plan.sampler_seed,
        steps=full_pixel_plan.steps,
        spacetime_schedule_sha256=full_pixel_plan.spacetime_schedule_sha256,
        pixel_chunk_manifest_sha256=pixel_manifest.hexdigest(),
        sample_schedule_sha256=workload_receipt.sample_schedule_sha256,
        sampled_image_count=full_pixel_plan.sampled_image_count,
        pixel_chunk_count=chunk_count,
        target_pixels=target_pixels,
        maximum_pixels_per_chunk=maximum_chunk,
        heldout_maximum_pixels_per_chunk=MAXIMUM_PIXELS_PER_CHUNK,
        selected_pixels_per_spacetime_sample=int(
            config["training_sampling"]["pixels_per_spacetime_sample"]
        ),
        selected_pixel_ids_by_step=by_step,
        training_loss_contract=dict(config["training_loss"]),
        workload_receipt=workload_receipt,
    )


__all__ = (
    "SelectedRayWorkPlan",
    "WORK_PLAN_KIND",
    "WORK_PLAN_SCHEMA_VERSION",
    "build_selected_ray_work_plan",
)
