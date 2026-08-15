"""Executable row boundary for the matched selected-ray WorldFoam G4-v2 ablation."""

from __future__ import annotations

import importlib.util
import math
import os
import re
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from paper_training_protocol import lpips_alex_asset_status
from worldfoam_g4_selected_ray_contract import (
    DEFAULT_CONFIG,
    REQUIRED_ROUTES,
    build_selected_ray_workload_receipt,
    file_sha256,
    load_selected_ray_contract,
)
from worldfoam_g4_selected_ray_work_plan import build_selected_ray_work_plan
from worldfoam_g4_v2_capability import CAPABILITY_PATH as V2_CAPABILITY_PATH
import worldfoam_native4d_public_quality_row as v1


ROOT = Path(__file__).resolve().parents[2]
LANE2 = ROOT / "research_experiments" / "world_foam_lane2"
MAXIMUM_MPS_WORKING_SET_BYTES = 2 * 1024**3
if str(LANE2) not in sys.path:
    sys.path.insert(0, str(LANE2))

from verify_worldfoam_public_quality_ablation_v2 import (  # noqa: E402
    ROW_KIND,
    validate_raw_row,
)


def _expected_output_path(config: Mapping[str, Any], request: v1.RowRequest) -> Path:
    return (
        ROOT
        / str(config["output_root"])
        / request.scene
        / f"seed_{request.seed}"
        / request.route
        / "g4_v2_row.json"
    ).resolve()


def _base_resolution_request(
    *,
    base: Mapping[str, Any],
    base_path: Path,
    request: v1.RowRequest,
) -> v1.RowRequest:
    output = (
        ROOT
        / str(base["output_root"])
        / request.scene
        / f"seed_{request.seed}"
        / request.route
        / "g4_row.json"
    )
    return v1.RowRequest(
        config_path=base_path,
        protocol_path=request.protocol_path,
        scene=request.scene,
        seed=request.seed,
        route=request.route,
        output_path=output,
        allow_local_mps_execution=request.allow_local_mps_execution,
        dataset_capability_path=request.dataset_capability_path,
    )


def _required_source_capability(config_path: Path) -> dict[str, Any]:
    from worldfoam_g4_v2_capability import required_source_capability

    return required_source_capability(config_path)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = Path(path).resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError("no existing parent for G4-v2 output")
        candidate = candidate.parent
    return candidate


def resolve_v2_context(
    request: v1.RowRequest,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    Any,
    v1.RowContext,
]:
    config_path = Path(request.config_path).resolve()
    config, base, base_path = load_selected_ray_contract(config_path)
    if (
        request.scene not in config["matrix"]["scene_order"]
        or request.seed not in config["matrix"]["seed_order"]
        or request.route not in REQUIRED_ROUTES
    ):
        raise ValueError("v2 row request left the frozen matrix")
    if Path(request.output_path).resolve() != _expected_output_path(config, request):
        raise ValueError("v2 output path differs from the frozen matrix")
    (
        base_config,
        base_receipt,
        protocol,
        route_spec,
        scene_receipt,
        full_plan,
    ) = v1.resolve_row_request(
        _base_resolution_request(base=base, base_path=base_path, request=request)
    )
    if base_config != base:
        raise ValueError("v2 base config changed during v1 resolution")
    workload = build_selected_ray_workload_receipt(
        config=config,
        base=base,
        config_path=config_path,
        base_path=base_path,
        scene=request.scene,
        seed=request.seed,
    )
    work_plan = build_selected_ray_work_plan(
        config=config,
        protocol=protocol,
        seed=request.seed,
        full_pixel_plan=full_plan,
        workload_receipt=workload,
    )
    capability_path = request.dataset_capability_path or v1.default_dataset_capability_path(
        protocol
    )
    dataset_capability = v1.load_dataset_capability(
        capability_path,
        request=request,
        protocol=protocol,
        scene_receipt=scene_receipt,
    )
    source = v1._source_identity()
    context = v1.RowContext(
        request=request,
        config=base,
        config_receipt=base_receipt,
        protocol=protocol,
        route_spec=route_spec,
        scene_receipt=scene_receipt,
        work_plan=work_plan,  # type: ignore[arg-type]
        source_commit=str(source["repository_commit"]),
        dataset_capability=dataset_capability,
    )
    return config, base, base_path, workload, context


def preflight_v2(
    request: v1.RowRequest,
    *,
    sample_host_resources: bool = False,
) -> tuple[list[str], dict[str, Any], tuple[Any, ...] | None]:
    blockers: list[str] = []
    details: dict[str, Any] = {"allocation_started": False}
    try:
        resolved = resolve_v2_context(request)
    except Exception as error:
        return [f"v2_request_invalid:{type(error).__name__}:{error}"], details, None
    config, base, base_path, workload, context = resolved
    details["work_plan"] = context.work_plan.as_dict()
    details["workload_receipt"] = workload.as_dict()
    if request.route.startswith("worldfoam_"):
        disk_root = _nearest_existing_parent(request.output_path.parent)
        required_disk_bytes = int(
            config["execution"]["minimum_free_disk_bytes_before_worldfoam_row"]
        )
        disk_receipt = {
            "filesystem_probe": str(disk_root),
            "sampled": bool(sample_host_resources),
            "free_bytes": None,
            "required_free_bytes": required_disk_bytes,
            "prediction_spool_bytes": 707_788_800,
            "target_spool_bytes": 176_947_200,
            "total_spool_bytes": 884_736_000,
            "passed": None,
        }
        if sample_host_resources:
            free_disk_bytes = int(shutil.disk_usage(disk_root).free)
            disk_receipt.update(
                {
                    "free_bytes": free_disk_bytes,
                    "passed": free_disk_bytes >= required_disk_bytes,
                }
            )
            if free_disk_bytes < required_disk_bytes:
                blockers.append("worldfoam_heldout_spool_disk_space_insufficient")
        details["worldfoam_heldout_spool_disk_preflight"] = disk_receipt
    if not request.allow_local_mps_execution:
        blockers.append("local_mps_execution_not_acknowledged")
    if sys.platform != "darwin":
        blockers.append("metal_g4_requires_macos")
    initialization_blockers, initialization = v1._initialization_blockers(
        context.scene_receipt["initialization"]
    )
    blockers.extend(initialization_blockers)
    details["initialization"] = initialization
    source = v1._source_identity()
    details["source"] = source
    if source.get("repository_dirty") is not False:
        blockers.append("paper_evidence_requires_clean_source")
    if re.fullmatch(r"[0-9a-f]{40}", str(source.get("repository_commit", ""))) is None:
        blockers.append("source_commit_invalid")
    if not V2_CAPABILITY_PATH.is_file():
        blockers.append("g4_v2_source_capability_missing")
    else:
        try:
            capability = v1._load_json(V2_CAPABILITY_PATH)
        except Exception as error:
            blockers.append(f"g4_v2_source_capability_invalid:{type(error).__name__}:{error}")
        else:
            if capability != _required_source_capability(Path(request.config_path)):
                blockers.append("g4_v2_source_capability_stale")
    from run_worldfoam_public_quality_ablation_v2 import _native_binary_blockers

    blockers.extend(_native_binary_blockers(base))
    executor_module = v1.ROUTE_EXECUTOR_MODULES[request.route]
    spec = importlib.util.find_spec(executor_module)
    if spec is None or spec.origin is None or not Path(spec.origin).is_file():
        blockers.append(f"production_route_executor_missing:{executor_module}")
    lpips = lpips_alex_asset_status()
    details["lpips_assets"] = lpips
    if lpips.get("status") != "pass":
        blockers.append("paper_lpips_assets_missing_or_drifted")
    if shutil.which("ffmpeg") is None:
        blockers.append("ffmpeg_media_writer_missing")
    if importlib.util.find_spec("wandb") is None:
        blockers.append("wandb_run_writer_missing")
    details["runtime_ready"] = not blockers
    return sorted(set(blockers)), details, resolved


def _configure_mps_working_set_limit(
    torch: Any,
    maximum_bytes: int,
) -> dict[str, Any]:
    """Install and attest the hard MPS ceiling after static preflight."""

    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes < 1
        or maximum_bytes > MAXIMUM_MPS_WORKING_SET_BYTES
    ):
        raise ValueError("MPS working-set limit exceeds the fail-closed 2 GiB ceiling")
    if not torch.backends.mps.is_available():
        raise RuntimeError("G4-v2 row execution requires an available MPS backend")
    setter = getattr(torch.mps, "set_per_process_memory_fraction", None)
    recommended = getattr(torch.mps, "recommended_max_memory", None)
    if not callable(setter) or not callable(recommended):
        raise RuntimeError("PyTorch lacks the required MPS memory-limit API")
    recommended_bytes = int(recommended())
    if recommended_bytes < 1:
        raise RuntimeError("MPS recommended working set must be positive")
    effective_fraction = min(1.0, float(maximum_bytes) / float(recommended_bytes))
    if not 0.0 < effective_fraction <= 1.0:
        raise RuntimeError("effective MPS memory fraction is outside the safe range")
    setter(effective_fraction)
    payload = {
        "schema_version": 1,
        "kind": "worldfoam-g4-v2-row-mps-working-set-limit-v1",
        "requested_working_set_limit_bytes": maximum_bytes,
        "recommended_max_memory_bytes": recommended_bytes,
        "effective_fraction": effective_fraction,
        "effective_working_set_limit_bytes": min(maximum_bytes, recommended_bytes),
        "installed_before_dataset_executor_native_or_tensor_allocation": True,
    }
    return {**payload, "generation_digest": v1._sha256(payload)}


def _checkpoint_identity(
    context: v1.RowContext,
    checkpoint_path: Path,
) -> dict[str, Any]:
    from checkpoint_utils import load_checkpoint_mapping
    from worldfoam_g4_selected_ray_contract import canonical_sha256

    payload = load_checkpoint_mapping(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        label="G4-v2 final checkpoint",
    )
    expected = dict(context.work_plan.training_loss_contract)
    workload = context.work_plan.workload_receipt
    if (
        payload.get("training_loss_contract") != expected
        or payload.get("sample_schedule_sha256")
        != context.work_plan.sample_schedule_sha256
        or payload.get("v2_config_sha256") != workload.v2_config_sha256
        or payload.get("workload_receipt_generation_digest")
        != workload.generation_digest
        or payload.get("route_schedule_sha256") != workload.route_schedule_sha256
    ):
        raise ValueError("final checkpoint did not bind the complete v2 workload")
    return v1._file_identity(
        checkpoint_path,
        step=context.protocol.steps,
        training_loss_contract_sha256=canonical_sha256(expected),
        sample_schedule_sha256=context.work_plan.sample_schedule_sha256,
        v2_config_sha256=workload.v2_config_sha256,
        workload_receipt_generation_digest=workload.generation_digest,
        route_schedule_sha256=workload.route_schedule_sha256,
    )


def _run_v2_training_lifecycle(
    context: v1.RowContext,
    *,
    dataset: Any,
    session: Any,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Consume each selected target once, respecting executor I/O ownership."""

    internal_ingest = getattr(session, "accumulate_train_request", None)
    external_read_calls = 0
    external_observations = 0
    consumed_chunks = 0
    consumed_images: set[tuple[int, int]] = set()
    for work in context.work_plan.steps:
        session.begin_step(work)
        for request in context.work_plan.iter_step_training_chunks(work):
            if callable(internal_ingest):
                internal_ingest(request)
            else:
                payload = v1._validate_pixel_payload(
                    dataset.read_train_chunk(request),
                    request=request,
                )
                session.accumulate_train_chunk(request, payload)
                receipt = v1._fields(
                    payload.selected_read_receipt,
                    name="selected train read receipt",
                )
                external_read_calls += 1
                external_observations += int(receipt["observation_count"])
                del payload
            consumed_chunks += 1
            consumed_images.add((int(request.step), int(request.sample_slot)))
        session.finish_step(work)
    if (
        consumed_chunks != context.work_plan.pixel_chunk_count
        or len(consumed_images) != context.work_plan.sampled_image_count
    ):
        raise ArithmeticError("G4-v2 training lifecycle lost selected work")
    training = v1.validate_training_receipt(
        session.finalize_training(checkpoint_path),
        context=context,
        checkpoint_path=checkpoint_path,
    )
    if callable(internal_ingest):
        source_receipt_factory = getattr(session, "training_source_read_receipt", None)
        if not callable(source_receipt_factory):
            raise TypeError("internal target owner did not expose a source-read receipt")
        source_receipt = dict(source_receipt_factory())
    else:
        payload = {
            "schema_version": 1,
            "kind": "row-worker-external-selected-target-reads-v1",
            "ownership": "row_worker_external_single_read",
            "selected_pixel_read_call_count": external_read_calls,
            "selected_pixel_read_observation_count": external_observations,
            "full_frame_target_materialization_count": 0,
            "external_row_worker_target_read_call_count": external_read_calls,
            "request_schedule_sha256": context.work_plan.sample_schedule_sha256,
        }
        source_receipt = {**payload, "generation_digest": v1._sha256(payload)}
    if (
        source_receipt.get("selected_pixel_read_observation_count")
        != context.work_plan.target_pixels
        or source_receipt.get("full_frame_target_materialization_count") != 0
    ):
        raise ArithmeticError("G4-v2 target source was not read exactly once")
    return training, source_receipt


def _assemble_v2_row(
    *,
    context: v1.RowContext,
    config: Mapping[str, Any],
    base_path: Path,
    workload: Any,
    executor_capability: Mapping[str, Any],
    training: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    evaluation: v1.HeldoutEvaluationReceipt,
    heldout_media: Mapping[str, Any],
    wandb_run_file: Mapping[str, Any],
    row_measurements: Mapping[str, Any],
    target_source_read_receipt: Mapping[str, Any],
    heldout_execution_receipt: Mapping[str, Any],
    mps_working_set_limit_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    row = v1.assemble_raw_row(
        context,
        executor_capability=executor_capability,
        training=training,
        checkpoint=checkpoint,
        evaluation=evaluation,
        heldout_media=heldout_media,
        wandb_run_file=wandb_run_file,
        row_measurements=row_measurements,
    )
    row.update(
        {
            "schema_version": 2,
            "row_kind": ROW_KIND,
            "v2_config_path": str(Path(context.request.config_path).resolve().relative_to(ROOT)),
            "v2_config_sha256": workload.v2_config_sha256,
            "base_g4_v1_sha256": file_sha256(base_path),
            "training_sampling_kind": config["training_sampling"]["kind"],
            "training_loss_contract": dict(config["training_loss"]),
            "training_loss_contract_sha256": workload.training_loss_contract_sha256,
            "selected_pixels_per_spacetime_sample": (
                workload.selected_pixels_per_spacetime_sample
            ),
            "selected_loss_scalar_count": workload.selected_loss_scalar_count,
            "route_schedule_sha256": workload.route_schedule_sha256,
            "workload_receipt": workload.as_dict(),
            "workload_receipt_generation_digest": workload.generation_digest,
            "full_heldout_target_pixels": workload.heldout_target_pixels,
            "training_rasterized_work_claimed_equal": False,
            "target_source_read_receipt": dict(target_source_read_receipt),
            "heldout_execution_receipt": dict(heldout_execution_receipt),
            "mps_working_set_limit_receipt": dict(
                mps_working_set_limit_receipt
            ),
            "parent_rusage_memory_scope": (
                "worker_parent_only_excludes_children_use_process_group_watchdog"
            ),
            "heldout_wall_time_cross_route_comparable": False,
        }
    )
    return row


def _publish_v2_row(
    *,
    context: v1.RowContext,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    base_path: Path,
    workload: Any,
    row: Mapping[str, Any],
) -> Path:
    output = Path(context.request.output_path).resolve()
    if output.exists():
        raise FileExistsError(f"G4-v2 row already exists: {output}")
    errors = validate_raw_row(
        row,
        config=config,
        base=base,
        base_path=base_path,
        config_path=Path(context.request.config_path),
        workload=workload,
        source_commit=context.source_commit,
    )
    if errors:
        raise ValueError("raw G4-v2 row failed validation: " + "; ".join(errors))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.verify")
    temporary.unlink(missing_ok=True)
    try:
        v1._atomic_write_json(temporary, row)
        if v1._load_json(temporary) != dict(row):
            raise ValueError("serialized G4-v2 row differs from its validated object")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def execute_v2_row(
    request: v1.RowRequest,
    *,
    wandb_mode: str = "offline",
    maximum_mps_working_set_bytes: int = MAXIMUM_MPS_WORKING_SET_BYTES,
) -> dict[str, Any]:
    blockers, details, resolved = preflight_v2(
        request,
        sample_host_resources=True,
    )
    if blockers or resolved is None:
        raise RuntimeError("G4-v2 row aborted before allocation: " + ", ".join(blockers))
    config, base, base_path, workload, context = resolved
    dataset = session = media_sink = row_memory_sampler = None
    try:
        row_started = time.perf_counter()
        import torch
        from device_memory import DeviceMemorySampler

        mps_limit_receipt = _configure_mps_working_set_limit(
            torch,
            maximum_mps_working_set_bytes,
        )
        setup_started = time.perf_counter()
        row_memory_sampler = DeviceMemorySampler(torch.device("mps"))
        row_memory_sampler.start()
        dataset = v1.load_public_quality_dataset(context)
        executor = v1._load_route_executor(context)
        capability = v1._validate_executor_capability(
            executor.capability(context), context=context
        )
        if request.route.startswith("worldfoam_"):
            v1.validate_worldfoam_training_inputs(dataset, context=context)
        session = executor.open_session(context, dataset)
        setup_elapsed = time.perf_counter() - setup_started
        checkpoint_path = request.output_path.parent / "checkpoint_final.pt"
        training, target_source_read_receipt = _run_v2_training_lifecycle(
            context,
            dataset=dataset,
            session=session,
            checkpoint_path=checkpoint_path,
        )
        checkpoint = _checkpoint_identity(context, checkpoint_path)
        media_sink = v1.StreamingSideBySideMp4Sink(
            request.output_path.parent / "heldout_full_temporal.mp4",
            height=context.protocol.final_stage.image_size.height,
            width=context.protocol.final_stage.image_size.width,
            fps=context.protocol.dataset.fps,
        )
        evaluation_started = time.perf_counter()
        if request.route.startswith("worldfoam_"):
            from worldfoam_spatial_major_heldout_evaluator import (
                evaluate_worldfoam_spatial_major_final_checkpoint,
            )

            spatial_evaluation = evaluate_worldfoam_spatial_major_final_checkpoint(
                context,
                session=session,
                media_sink=media_sink,
                maximum_render_call_count=int(
                    config["tractability_limits"][
                        "maximum_heldout_spatial_major_render_call_count"
                    ]
                ),
                spool_directory=request.output_path.parent,
            )
            evaluation = spatial_evaluation.evaluation
            heldout_execution_receipt = dict(
                spatial_evaluation.spatial_replay_receipt
            )
        else:
            evaluation = v1.evaluate_final_checkpoint(
                context,
                dataset=dataset,
                session=session,
                media_sink=media_sink,
            )
        evaluation_elapsed = time.perf_counter() - evaluation_started
        media_sink = None
        if evaluation.pixel_count != workload.heldout_target_pixels:
            raise ArithmeticError("G4-v2 heldout evaluator changed full coverage")
        if not request.route.startswith("worldfoam_"):
            heldout_payload = {
                "schema_version": 1,
                "kind": "gaussian-frame-major-full-image-heldout-v1",
                "camera_count": len(context.protocol.dataset.heldout_cameras),
                "frame_count": evaluation.frame_count,
                "target_pixel_count": evaluation.pixel_count,
                "pixel_chunk_count": evaluation.pixel_chunk_count,
                "coverage_sha256": evaluation.coverage_sha256,
                "full_pixel_full_temporal": True,
            }
            heldout_execution_receipt = {
                **heldout_payload,
                "generation_digest": v1._sha256(heldout_payload),
            }
        row_memory_sampler.stop()
        memory = row_memory_sampler.stats()
        row_measurements = {
            "process_lifetime_peak_rss_through_heldout_evaluation_bytes": (
                v1._process_lifetime_peak_rss_bytes()
            ),
            "sampled_peak_mps_driver_through_heldout_evaluation_bytes": max(
                int(memory["sampled_peak_driver_allocated_bytes"]),
                int(training["sampled_peak_mps_driver_during_training_and_checkpoint_bytes"]),
            ),
            "executor_dataset_and_model_setup_elapsed_s": float(setup_elapsed),
            "heldout_evaluation_elapsed_s": float(evaluation_elapsed),
            "full_row_through_heldout_evaluation_elapsed_s": float(
                time.perf_counter() - row_started
            ),
        }
        heldout_media = v1._file_identity(
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
            "final_active_primitive_count_per_render": 1024,
            "stored_primitive_state_count": 307200 if request.route == "dynamic_3dgs" else 1024,
            "process_lifetime_peak_rss_through_checkpoint_bytes": int(
                training["process_lifetime_peak_rss_through_checkpoint_bytes"]
            ),
            "sampled_peak_mps_driver_during_training_and_checkpoint_bytes": int(
                training["sampled_peak_mps_driver_during_training_and_checkpoint_bytes"]
            ),
            "training_and_checkpoint_elapsed_s": float(
                training["training_and_checkpoint_elapsed_s"]
            ),
            **row_measurements,
        }
        wandb_receipt = v1.write_wandb_run_file(
            context,
            metrics=evaluation.metrics,
            cost=preliminary_cost,
            media_path=evaluation.media_path,
            mode=wandb_mode,
        )
        if v1._source_identity() != {
            "repository_commit": context.source_commit,
            "repository_dirty": False,
        }:
            raise RuntimeError("source changed while executing G4-v2 row")
        row = _assemble_v2_row(
            context=context,
            config=config,
            base_path=base_path,
            workload=workload,
            executor_capability=capability,
            training=training,
            checkpoint=checkpoint,
            evaluation=evaluation,
            heldout_media=heldout_media,
            wandb_run_file=wandb_receipt,
            row_measurements=row_measurements,
            target_source_read_receipt=target_source_read_receipt,
            heldout_execution_receipt=heldout_execution_receipt,
            mps_working_set_limit_receipt=mps_limit_receipt,
        )
        output = _publish_v2_row(
            context=context,
            config=config,
            base=base,
            base_path=base_path,
            workload=workload,
            row=row,
        )
        return {
            "status": "measured",
            "row": str(output.relative_to(ROOT)),
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


def build_v2_row_plan(request: v1.RowRequest) -> dict[str, Any]:
    blockers, details, _resolved = preflight_v2(request)
    payload = {
        "schema_version": 2,
        "kind": "worldfoam-native4d-public-quality-selected-ray-row-plan-v2",
        "scene": request.scene,
        "seed": request.seed,
        "route": request.route,
        "output": str(Path(request.output_path).resolve().relative_to(ROOT)),
        "runtime_ready": not blockers,
        "allocation_started": False,
        "blockers": blockers,
        "details": details,
    }
    return {**payload, "plan_sha256": v1._sha256(payload)}


__all__ = (
    "MAXIMUM_MPS_WORKING_SET_BYTES",
    "V2_CAPABILITY_PATH",
    "build_v2_row_plan",
    "execute_v2_row",
    "preflight_v2",
    "resolve_v2_context",
)
