"""Shared production lifecycle for the G4 Gaussian comparison routes.

The public-quality row worker owns the sample schedule, target/ray streaming,
held-out evaluator, and artifact receipts.  This module adapts that lifecycle
to image rasterizers without materializing a target video: one sampled image
is rasterized once, its loss is accumulated from the worker's bounded pixel
chunks, and that image's graph is released before the next sample is rendered.

Route modules provide the actual trainable representation and native renderer.
No fake-native, procedural-target, reduced-pixel, or CPU paper-evidence path is
implemented here.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from worldfoam_native4d_public_quality_row import (
    PixelChunkPayload,
    PixelChunkRequest,
    RowContext,
    StepWork,
)
from public_quality_runtime_smoke import (
    RUNTIME_SMOKE_KIND,
    RUNTIME_SMOKE_STATUS,
    canonical_sha256,
    validate_public_quality_runtime_smoke,
)


REPRESENTATION_SEED_KEYS = frozenset(
    {
        "positions0_f32_cpu",
        "colors_f32_cpu",
        "source_frame_indices_i64_cpu",
        "source_time_provenance",
        "initializer_generation_digest",
        "material_seed_generation_digest",
        "sites_content_digest",
        "material_content_digest",
        "representation_seed_content_sha256",
    }
)
_SEED_TENSOR_KEYS = (
    "positions0_f32_cpu",
    "colors_f32_cpu",
    "source_frame_indices_i64_cpu",
)
_SEED_DIGEST_KEYS = (
    "initializer_generation_digest",
    "material_seed_generation_digest",
    "sites_content_digest",
    "material_content_digest",
    "representation_seed_content_sha256",
)
ROBUST_L1_EPSILON = 1.0e-3


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fields(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    result = getattr(value, "__dict__", None)
    if not isinstance(result, Mapping):
        raise TypeError(f"{name} must expose mapping or dataclass fields")
    return result


def _maximum_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _synchronize(device: Any) -> None:
    import torch

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _to_cpu_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, Mapping):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    return value


def model_state_sha256(model: Any) -> str:
    """Content-bind a state dict while staging one tensor at a time on CPU."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        metadata = {
            "name": str(name),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "bytes": int(value.numel() * value.element_size()),
        }
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
        digest.update(memoryview(value.numpy()).cast("B"))
        digest.update(b"\n")
    return digest.hexdigest()


def _tensor_content_sha256(name: str, tensor: Any) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "name": str(name),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "bytes": int(value.numel() * value.element_size()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def representation_seed_content_sha256(seed: Mapping[str, Any]) -> str:
    """Return a portable identity for actual seed tensors and time metadata."""

    return _sha256_json(
        {
            "schema_version": 1,
            "source_time_provenance": str(seed["source_time_provenance"]),
            "tensor_sha256": {
                key: _tensor_content_sha256(key, seed[key])
                for key in _SEED_TENSOR_KEYS
            },
        }
    )


def load_fresh_representation_seed(
    dataset: Any,
    *,
    expected_site_count: int,
) -> dict[str, Any]:
    """Load and verify two independent clones of the sealed common seed."""

    import torch

    factory = getattr(dataset, "representation_seed", None)
    if not callable(factory):
        raise TypeError(
            "public Gaussian routes require dataset.representation_seed()"
        )

    def validate(raw: Any) -> dict[str, Any]:
        value = dict(_fields(raw, name="representation seed"))
        if set(value) != set(REPRESENTATION_SEED_KEYS):
            raise ValueError("representation seed keys changed")
        positions = value["positions0_f32_cpu"]
        colors = value["colors_f32_cpu"]
        source_frames = value["source_frame_indices_i64_cpu"]
        for name, tensor in (("positions", positions), ("colors", colors)):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != (int(expected_site_count), 3)
                or not tensor.is_contiguous()
                or tensor.requires_grad
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise ValueError(
                    f"representation seed {name} changed dtype, device, shape, or finiteness"
                )
        if bool(torch.any((colors < 0.0) | (colors > 1.0)).item()):
            raise ValueError("representation seed colors must lie in [0,1]")
        if (
            not isinstance(source_frames, torch.Tensor)
            or source_frames.device.type != "cpu"
            or source_frames.dtype != torch.int64
            or tuple(source_frames.shape) != (int(expected_site_count),)
            or not source_frames.is_contiguous()
            or source_frames.requires_grad
            or bool(torch.any(source_frames < -1).item())
        ):
            raise ValueError(
                "representation seed source frames changed dtype, device, shape, or sentinel policy"
            )
        provenance = value["source_time_provenance"]
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("representation seed source-time provenance is missing")
        if any(not _valid_sha256(value[key]) for key in _SEED_DIGEST_KEYS):
            raise ValueError("representation seed contains an invalid generation digest")
        if value["representation_seed_content_sha256"] != representation_seed_content_sha256(
            value
        ):
            raise ValueError("representation seed content digest disagrees with its tensors")
        return value

    first = validate(factory())
    second = validate(factory())
    for tensor_key in _SEED_TENSOR_KEYS:
        if first[tensor_key].data_ptr() == second[tensor_key].data_ptr():
            raise ValueError("representation_seed() returned aliased mutable tensors")
        if not bool(torch.equal(first[tensor_key], second[tensor_key])):
            raise ValueError("representation_seed() is not deterministic")
    if any(first[key] != second[key] for key in _SEED_DIGEST_KEYS):
        raise ValueError("representation seed generation changed between fresh calls")
    return first


def representation_seed_identity(seed: Mapping[str, Any]) -> dict[str, str]:
    return {
        **{key: str(seed[key]) for key in _SEED_DIGEST_KEYS},
        "source_time_provenance": str(seed["source_time_provenance"]),
    }


def camera_to_device(camera: Any, device: Any) -> Any:
    import torch
    from camera import CameraSpec

    if not isinstance(camera, CameraSpec):
        raise TypeError("dataset.camera_spec() must return CameraSpec")
    matrix = camera.camera_to_world
    if (
        not isinstance(matrix, torch.Tensor)
        or tuple(matrix.shape) != (4, 4)
        or not bool(torch.isfinite(matrix).all().item())
    ):
        raise ValueError("public camera has an invalid camera_to_world matrix")

    def scalar(value: Any, *, name: str) -> float:
        result = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
        if not math.isfinite(result):
            raise ValueError(f"public camera {name} is non-finite")
        return result

    fx = scalar(camera.fx, name="fx")
    fy = scalar(camera.fy, name="fy")
    cx = scalar(camera.cx, name="cx")
    cy = scalar(camera.cy, name="cy")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("public camera focal lengths must be positive")
    distortion = camera.distortion
    if distortion is not None:
        distortion = torch.as_tensor(
            distortion,
            dtype=torch.float32,
            device=device,
        ).contiguous()
        if not bool(torch.isfinite(distortion).all().item()):
            raise ValueError("public camera distortion is non-finite")
    return CameraSpec(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_to_world=matrix.detach().to(
            device=device,
            dtype=torch.float32,
        ).contiguous(),
        lens_model=camera.lens_model,
        distortion=distortion,
    )


def calibrated_camera_for_chunk(
    dataset: Any,
    request: PixelChunkRequest,
    rays_f32_cpu: Any,
    *,
    device: Any,
) -> Any:
    """Fetch a camera and bind it back to deterministic rays from this chunk."""

    import torch
    from camera import build_camera_rays_at_pixels

    factory = getattr(dataset, "camera_spec", None)
    if not callable(factory):
        raise TypeError("public Gaussian routes require dataset.camera_spec()")
    camera_cpu = camera_to_device(
        factory(
            split=request.split,
            camera_index=request.camera_index,
            frame_index=request.frame_index,
        ),
        torch.device("cpu"),
    )
    local_indices = (
        list(range(request.pixel_count))
        if request.pixel_ids is not None
        else sorted({0, request.pixel_count // 2, request.pixel_count - 1})
    )
    local = torch.tensor(local_indices, dtype=torch.long, device="cpu")
    explicit_pixels = request.pixel_ids
    pixels = (
        local + int(request.pixel_start)
        if explicit_pixels is None
        else torch.tensor(
            tuple(int(explicit_pixels[index]) for index in local_indices),
            dtype=torch.long,
            device="cpu",
        )
    )
    origins, directions = build_camera_rays_at_pixels(
        camera_cpu,
        pixels,
        height=request.image_height,
        width=request.image_width,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = torch.cat((origins, directions), dim=-1).contiguous()
    observed = rays_f32_cpu.index_select(0, local)
    if not bool(torch.allclose(expected, observed, rtol=3.0e-5, atol=3.0e-5)):
        raise ValueError("dataset camera calibration disagrees with its sealed rays")
    return camera_to_device(camera_cpu, device)


class GaussianPublicQualitySession:
    """Exact chunk consumer shared by the two production Gaussian routes."""

    def __init__(
        self,
        *,
        context: RowContext,
        dataset: Any,
        model: Any,
        optimizer: Any,
        device: Any,
        seed_identity: Mapping[str, str],
        route_contract: Mapping[str, Any],
        base_lr: float,
        render_image: Callable[[PixelChunkRequest, Any], Any],
        regularization: Callable[[], Any],
        set_active_count: Callable[[int], None],
    ) -> None:
        import torch
        from device_memory import DeviceMemorySampler

        if device.type != "mps" or not torch.backends.mps.is_available():
            raise RuntimeError("G4 Gaussian production routes require available MPS")
        self.context = context
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.seed_identity = dict(seed_identity)
        self.route_contract = dict(route_contract)
        self.base_lr = float(base_lr)
        self._render_image = render_image
        self._regularization = regularization
        self._set_active_count = set_active_count
        self._memory_sampler = DeviceMemorySampler(device)
        self._started_at: float | None = None
        self._active_work: StepWork | None = None
        self._active_sample_slot: int | None = None
        self._active_prediction: Any = None
        self._active_sample_pixels = 0
        self._active_sample_loss_sum: Any = None
        self._step_sample_count = 0
        self._step_backward_passes = 0
        self._optimizer_steps = 0
        self._target_pixels = 0
        self._sampled_images = 0
        self._pixel_chunks = 0
        self._rasterized_pixels = 0
        self._finalized = False
        self._closed = False
        self._heldout_key: tuple[int, int] | None = None
        self._heldout_frame: Any = None
        self._heldout_next_pixel = 0
        self._training_loss_contract = dict(
            getattr(
                context.work_plan,
                "training_loss_contract",
                {
                    "identifier": "robust_l1_charbonnier_eps_1e-3_v1",
                    "formula": "mean(sqrt((prediction-target)^2+1e-6))",
                    "normalization": "mean_over_selected_rgb_scalars",
                },
            )
        )
        if self._training_loss_contract.get("identifier") not in {
            "robust_l1_charbonnier_eps_1e-3_v1",
            "rgb_mse_mean_v1",
        }:
            raise ValueError("Gaussian public-quality loss contract is unsupported")

    def begin_step(self, work: StepWork) -> None:
        if self._closed or self._finalized:
            raise RuntimeError("cannot begin a step on a finalized Gaussian session")
        if self._active_work is not None:
            raise RuntimeError("previous Gaussian optimizer step is still active")
        if work.step != self._optimizer_steps:
            raise ValueError("Gaussian optimizer steps must be contiguous and ordered")
        if self._started_at is None:
            if work.step != 0:
                raise ValueError("Gaussian timing must begin at optimizer step zero")
            self._memory_sampler.start()
            self._started_at = time.perf_counter()
        self._set_active_count(work.stage.primitive_count)
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * float(work.stage.lr_multiplier)
        self.optimizer.zero_grad(set_to_none=True)
        self._active_work = work
        self._active_sample_slot = None
        self._active_prediction = None
        self._active_sample_pixels = 0
        self._active_sample_loss_sum = None
        self._step_sample_count = 0
        self._step_backward_passes = 0

    def _finish_active_sample(self) -> None:
        import torch

        if self._active_sample_slot is None:
            return
        assert self._active_work is not None
        expected_pixels = int(
            getattr(
                self.context.work_plan,
                "selected_pixels_per_spacetime_sample",
                self._active_work.stage.image_size.pixels,
            )
        )
        if self._active_sample_pixels != expected_pixels:
            raise ArithmeticError("Gaussian sampled image target coverage changed")
        if self._active_sample_loss_sum is None:
            raise ArithmeticError("Gaussian sampled image accumulated no reconstruction loss")
        denominator = float(
            len(self._active_work.batch.samples)
            * expected_pixels
            * 3
        )
        reconstruction = self._active_sample_loss_sum / denominator
        if not bool(torch.isfinite(reconstruction.detach()).all().item()):
            raise FloatingPointError("Gaussian public-quality reconstruction became non-finite")
        reconstruction.backward()
        self._step_backward_passes += 1
        self._active_prediction = None
        self._active_sample_loss_sum = None
        self._active_sample_pixels = 0

    def accumulate_train_chunk(
        self,
        request: PixelChunkRequest,
        payload: PixelChunkPayload,
    ) -> None:
        import torch

        work = self._active_work
        if work is None:
            raise RuntimeError("Gaussian train chunk arrived outside begin_step/finish_step")
        if request.split != "train" or request.step != work.step:
            raise ValueError("Gaussian train chunk belongs to a different step/split")
        if request.sample_slot is None or not 0 <= request.sample_slot < len(
            work.batch.samples
        ):
            raise ValueError("Gaussian train chunk has an invalid sample slot")
        sample = work.batch.samples[request.sample_slot]
        if (
            request.camera_index != sample.view_index
            or request.frame_index != sample.frame_index
            or request.image_height != work.stage.image_size.height
            or request.image_width != work.stage.image_size.width
        ):
            raise ValueError("Gaussian train chunk differs from the sealed sample batch")

        if request.sample_slot != self._active_sample_slot:
            self._finish_active_sample()
            if request.sample_slot != self._step_sample_count or request.pixel_start != 0:
                raise ValueError("Gaussian sample/chunk traversal is not canonical")
            camera = calibrated_camera_for_chunk(
                self.dataset,
                request,
                payload.rays_f32_cpu,
                device=self.device,
            )
            prediction = self._render_image(request, camera)
            if (
                not isinstance(prediction, torch.Tensor)
                or tuple(prediction.shape)
                != (request.image_height, request.image_width, 3)
                or prediction.device != self.device
                or not bool(torch.isfinite(prediction).all().item())
            ):
                raise ValueError("production Gaussian renderer returned an invalid image")
            self._active_prediction = prediction.reshape(-1, 3)
            self._active_sample_slot = request.sample_slot
            self._step_sample_count += 1
            self._sampled_images += 1
            self._rasterized_pixels += request.image_height * request.image_width
        else:
            calibrated_camera_for_chunk(
                self.dataset,
                request,
                payload.rays_f32_cpu,
                device=torch.device("cpu"),
            )

        if request.pixel_start != self._active_sample_pixels:
            raise ValueError("Gaussian train pixel chunks are missing, repeated, or reordered")
        prediction_chunk = (
            self._active_prediction[request.pixel_start : request.pixel_stop]
            if request.pixel_ids is None
            else self._active_prediction.index_select(
                0,
                torch.tensor(
                    request.pixel_ids,
                    dtype=torch.long,
                    device=self.device,
                ),
            )
        )
        target = payload.target_rgb_f32_cpu.to(
            device=self.device,
            dtype=torch.float32,
        )
        residual_square = (prediction_chunk - target).square()
        chunk_loss_sum = (
            residual_square.sum()
            if self._training_loss_contract["identifier"] == "rgb_mse_mean_v1"
            else torch.sqrt(residual_square + ROBUST_L1_EPSILON**2).sum()
        )
        self._active_sample_loss_sum = (
            chunk_loss_sum
            if self._active_sample_loss_sum is None
            else self._active_sample_loss_sum + chunk_loss_sum
        )
        self._active_sample_pixels += request.pixel_count
        self._target_pixels += request.pixel_count
        self._pixel_chunks += 1

    def finish_step(self, work: StepWork) -> None:
        import torch

        if self._active_work is not work:
            raise ValueError("Gaussian finish_step received a foreign StepWork")
        self._finish_active_sample()
        if (
            self._step_sample_count != len(work.batch.samples)
            or self._step_backward_passes != len(work.batch.samples)
        ):
            raise ArithmeticError("Gaussian optimizer step did not consume its full batch")
        regularization = self._regularization()
        if not isinstance(regularization, torch.Tensor) or not bool(
            torch.isfinite(regularization.detach()).all().item()
        ):
            raise FloatingPointError("Gaussian public-quality regularization became non-finite")
        if regularization.requires_grad:
            regularization.backward()
        elif float(regularization.detach().cpu().item()) != 0.0:
            raise RuntimeError("nonzero Gaussian regularization lost its gradient graph")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self._optimizer_steps += 1
        self._active_work = None
        self._active_sample_slot = None
        self._active_prediction = None
        self._active_sample_loss_sum = None
        self._step_backward_passes = 0

    def finalize_training(self, checkpoint_path: Path) -> Mapping[str, Any]:
        import torch
        from checkpoint_utils import atomic_torch_save

        if self._closed or self._finalized or self._active_work is not None:
            raise RuntimeError("Gaussian training cannot be finalized in its current state")
        if self._optimizer_steps != self.context.protocol.steps:
            raise ArithmeticError("Gaussian training ended before the frozen final step")
        if self._started_at is None:
            raise RuntimeError("Gaussian training timer never started")
        expected = self.context.work_plan
        expected_rasterized_pixels = sum(
            len(work.batch.samples) * work.stage.image_size.pixels
            for work in expected.steps
        )
        if (
            self._target_pixels != expected.target_pixels
            or self._sampled_images != expected.sampled_image_count
            or self._pixel_chunks != expected.pixel_chunk_count
            or self._rasterized_pixels != expected_rasterized_pixels
        ):
            raise ArithmeticError(
                "Gaussian training counters differ from the sealed work plan"
            )
        _synchronize(self.device)
        state_digest = model_state_sha256(self.model)
        representation_digest = _sha256_json(
            {
                "schema_version": 2,
                "route": self.context.request.route,
                "initialization_content_sha256": self.seed_identity[
                    "representation_seed_content_sha256"
                ],
                "model_state_sha256": state_digest,
            }
        )
        execution_environment_digest = _sha256_json(
            {
                "schema_version": 1,
                "source_commit": self.context.source_commit,
                "route_contract": self.route_contract,
            }
        )
        atomic_torch_save(
            {
                "schema_version": 1,
                "route": self.context.request.route,
                "step": self._optimizer_steps,
                "source_commit": self.context.source_commit,
                "sample_schedule_sha256": (
                    self.context.work_plan.sample_schedule_sha256
                ),
                "training_loss_contract": self._training_loss_contract,
                **(
                    {
                        "v2_config_sha256": (
                            self.context.work_plan.workload_receipt.v2_config_sha256
                        ),
                        "workload_receipt_generation_digest": (
                            self.context.work_plan.workload_receipt.generation_digest
                        ),
                        "route_schedule_sha256": (
                            self.context.work_plan.workload_receipt.route_schedule_sha256
                        ),
                    }
                    if getattr(self.context.work_plan, "workload_receipt", None)
                    is not None
                    else {}
                ),
                "seed_identity": self.seed_identity,
                "route_contract": self.route_contract,
                "model_state_sha256": state_digest,
                "representation_sha256": representation_digest,
                "execution_environment_sha256": execution_environment_digest,
                "model": _to_cpu_tree(self.model.state_dict()),
                "optimizer": _to_cpu_tree(self.optimizer.state_dict()),
            },
            checkpoint_path,
        )
        _synchronize(self.device)
        self._memory_sampler.stop()
        training_and_checkpoint_elapsed_s = time.perf_counter() - self._started_at
        memory = self._memory_sampler.stats()
        parameters = tuple(self.model.parameters())
        self._finalized = True
        return {
            "optimizer_steps": self._optimizer_steps,
            "target_pixels_consumed": self._target_pixels,
            "sampled_image_count": self._sampled_images,
            "pixel_chunk_count": self._pixel_chunks,
            "rasterized_pixels": self._rasterized_pixels,
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "parameter_bytes": sum(
                parameter.numel() * parameter.element_size()
                for parameter in parameters
            ),
            "process_lifetime_peak_rss_through_checkpoint_bytes": _maximum_rss_bytes(),
            "sampled_peak_mps_driver_during_training_and_checkpoint_bytes": int(
                memory["sampled_peak_driver_allocated_bytes"]
            ),
            "training_and_checkpoint_elapsed_s": float(
                training_and_checkpoint_elapsed_s
            ),
            "representation_sha256": representation_digest,
            "checkpoint_step": self._optimizer_steps,
        }

    def render_heldout_chunk(
        self,
        request: PixelChunkRequest,
        rays_f32_cpu: Any,
    ) -> Any:
        import torch

        if not self._finalized or self._closed or request.split != "heldout":
            raise RuntimeError("heldout Gaussian rendering requires a final checkpoint")
        key = (request.camera_index, request.frame_index)
        if key != self._heldout_key:
            if (
                self._heldout_key is not None
                and self._heldout_next_pixel != request.image_height * request.image_width
            ):
                raise ArithmeticError("heldout Gaussian frame traversal was incomplete")
            if request.pixel_start != 0:
                raise ValueError("heldout Gaussian frame must begin at pixel zero")
            camera = calibrated_camera_for_chunk(
                self.dataset,
                request,
                rays_f32_cpu,
                device=self.device,
            )
            with torch.no_grad():
                prediction = self._render_image(request, camera)
                if (
                    not isinstance(prediction, torch.Tensor)
                    or tuple(prediction.shape)
                    != (request.image_height, request.image_width, 3)
                    or not bool(torch.isfinite(prediction).all().item())
                ):
                    raise ValueError("heldout Gaussian renderer returned an invalid image")
                self._heldout_frame = prediction.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                ).reshape(-1, 3).contiguous()
            self._heldout_key = key
            self._heldout_next_pixel = 0
        else:
            calibrated_camera_for_chunk(
                self.dataset,
                request,
                rays_f32_cpu,
                device=torch.device("cpu"),
            )
        if request.pixel_start != self._heldout_next_pixel:
            raise ValueError("heldout Gaussian chunks are missing, repeated, or reordered")
        result = self._heldout_frame[request.pixel_start : request.pixel_stop]
        self._heldout_next_pixel = request.pixel_stop
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._memory_sampler.stop()
        self._active_prediction = None
        self._heldout_frame = None
        self._closed = True


def executor_capability(context: RowContext) -> dict[str, Any]:
    return {
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
        "compiled_shared_adjoint": False,
        "same_representation_framewise_replay": False,
        "proxy_or_test_artifact": False,
        "measurement_is_simulated": False,
    }


def run_gaussian_public_quality_runtime_smoke(
    *,
    context: RowContext,
    dataset: Any,
    executor: Any,
    executor_source: Path,
) -> dict[str, Any]:
    """Run one mapped public pixel through a real Gaussian native route.

    The helper deliberately does not use ``finalize_training`` or publish a G4
    row: production finalization requires all 300 optimizer steps.  Instead it
    performs one bounded update from the first sealed schedule sample, writes
    and reloads an ephemeral raw checkpoint, then renders one heldout pixel.
    The native rasterizers still evaluate one complete image; that fact is
    reported as ``rasterized_pixels`` rather than being disguised as one-pixel
    work.
    """

    import torch
    from checkpoint_utils import atomic_torch_save, load_checkpoint_mapping

    if context.request.route not in {"world_tubes", "dynamic_3dgs"}:
        raise ValueError("Gaussian runtime smoke received a non-Gaussian route")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Gaussian runtime smoke requires real MPS")
    attestation = dict(_fields(dataset.attestation(), name="dataset attestation"))
    if not (
        attestation.get("public_data") is True
        and attestation.get("calibrated_multiview") is True
        and attestation.get("procedural_target") is False
        and attestation.get("selected_pixel_reads") is True
    ):
        raise ValueError("Gaussian runtime smoke requires the sealed public dataset")

    started_at = time.perf_counter()
    capability = dict(executor.capability(context))
    session: GaussianPublicQualitySession | None = None
    try:
        session = executor.open_session(context, dataset)
        if session.device.type != "mps":
            raise RuntimeError("Gaussian runtime smoke opened a non-MPS session")
        session._memory_sampler.start()
        source_work = context.work_plan.steps[0]
        sample = source_work.batch.samples[0]
        height = int(source_work.stage.image_size.height)
        width = int(source_work.stage.image_size.width)
        pixel_index = (height // 2) * width + width // 2
        request = PixelChunkRequest(
            split="train",
            step=0,
            sample_slot=0,
            camera_index=int(sample.view_index),
            frame_index=int(sample.frame_index),
            pixel_start=pixel_index,
            pixel_count=1,
            image_height=height,
            image_width=width,
        )
        payload = dataset.read_train_chunk(request)
        target = payload.target_rgb_f32_cpu
        rays = payload.rays_f32_cpu
        if (
            not isinstance(target, torch.Tensor)
            or target.device.type != "cpu"
            or target.dtype != torch.float32
            or tuple(target.shape) != (1, 3)
            or not target.is_contiguous()
            or not bool(torch.isfinite(target).all().item())
            or float(target.min().item()) < 0.0
            or float(target.max().item()) > 1.0
            or not isinstance(rays, torch.Tensor)
            or rays.device.type != "cpu"
            or rays.dtype != torch.float32
            or tuple(rays.shape) != (1, 6)
            or not rays.is_contiguous()
            or not bool(torch.isfinite(rays).all().item())
        ):
            raise ValueError("Gaussian runtime smoke public train pixel is invalid")

        before = model_state_sha256(session.model)
        session._set_active_count(source_work.stage.primitive_count)
        for group in session.optimizer.param_groups:
            group["lr"] = session.base_lr * float(source_work.stage.lr_multiplier)
        session.optimizer.zero_grad(set_to_none=True)
        camera = calibrated_camera_for_chunk(
            dataset,
            request,
            rays,
            device=session.device,
        )
        prediction = session._render_image(request, camera)
        if (
            not isinstance(prediction, torch.Tensor)
            or prediction.device.type != "mps"
            or tuple(prediction.shape) != (height, width, 3)
            or not bool(torch.isfinite(prediction).all().item())
        ):
            raise ValueError("Gaussian runtime smoke native train render is invalid")
        predicted_pixel = prediction.reshape(-1, 3)[pixel_index : pixel_index + 1]
        target_mps = target.to(device=session.device, dtype=torch.float32)
        loss = torch.sqrt(
            (predicted_pixel - target_mps).square() + ROBUST_L1_EPSILON**2
        ).mean() + session._regularization()
        if not bool(torch.isfinite(loss.detach()).all().item()):
            raise FloatingPointError("Gaussian runtime smoke loss is non-finite")
        loss.backward()
        parameters = tuple(session.model.parameters())
        gradients = tuple(parameter.grad for parameter in parameters if parameter.grad is not None)
        finite_gradients = bool(gradients) and all(
            bool(torch.isfinite(gradient).all().item()) for gradient in gradients
        )
        nonzero_gradient = any(
            bool(torch.count_nonzero(gradient).item()) for gradient in gradients
        )
        if not finite_gradients or not nonzero_gradient:
            raise FloatingPointError(
                "Gaussian runtime smoke produced no finite nonzero parameter gradient"
            )
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        session.optimizer.step()
        _synchronize(session.device)
        after = model_state_sha256(session.model)
        if before == after:
            raise ArithmeticError("Gaussian runtime smoke observed no parameter update")

        # Exercise the exact serialization surface while keeping this receipt
        # explicitly outside the paper-evidence artifact tree.
        with tempfile.TemporaryDirectory(prefix="dynaworld-g4-runtime-smoke-") as temporary:
            checkpoint_path = Path(temporary) / "checkpoint.pt"
            atomic_torch_save(
                {
                    "schema_version": 1,
                    "route": context.request.route,
                    "step": 1,
                    "source_commit": context.source_commit,
                    "representation_sha256": after,
                    "model": _to_cpu_tree(session.model.state_dict()),
                    "optimizer": _to_cpu_tree(session.optimizer.state_dict()),
                },
                checkpoint_path,
            )
            loaded = load_checkpoint_mapping(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
                label="runtime-smoke checkpoint",
            )
            if (
                checkpoint_path.stat().st_size < 1
                or loaded.get("route") != context.request.route
                or loaded.get("step") != 1
                or loaded.get("representation_sha256") != after
                or not isinstance(loaded.get("model"), Mapping)
            ):
                raise ValueError("Gaussian runtime smoke checkpoint did not round-trip")

        heldout_request = PixelChunkRequest(
            split="heldout",
            step=None,
            sample_slot=None,
            camera_index=0,
            frame_index=int(sample.frame_index),
            pixel_start=pixel_index,
            pixel_count=1,
            image_height=height,
            image_width=width,
        )
        heldout_payload = dataset.read_heldout_chunk(heldout_request)
        heldout_camera = calibrated_camera_for_chunk(
            dataset,
            heldout_request,
            heldout_payload.rays_f32_cpu,
            device=session.device,
        )
        with torch.no_grad():
            heldout_image = session._render_image(heldout_request, heldout_camera)
            heldout_pixel = heldout_image.reshape(-1, 3)[
                pixel_index : pixel_index + 1
            ]
        finite_heldout = (
            tuple(heldout_image.shape) == (height, width, 3)
            and tuple(heldout_pixel.shape) == (1, 3)
            and bool(torch.isfinite(heldout_pixel).all().item())
        )
        if not finite_heldout:
            raise FloatingPointError("Gaussian runtime smoke heldout RGB is invalid")
        _synchronize(session.device)
        session._memory_sampler.stop()
        memory = session._memory_sampler.stats()

        route_contract = dict(session.route_contract)
        native_identity = route_contract.get("native_extension_identity")
        if not isinstance(native_identity, Mapping):
            raise TypeError("Gaussian runtime smoke lacks a native identity receipt")
        executor_source = Path(executor_source).resolve()
        if not executor_source.is_file():
            raise FileNotFoundError("Gaussian executor source disappeared during smoke")
        source_files = {
            key: value
            for key, value in route_contract.items()
            if key.endswith("_source_path") or key.endswith("_source_sha256")
        }
        source_receipt_sha256 = canonical_sha256(
            {
                "source_commit": context.source_commit,
                "executor_source_path": str(executor_source),
                "executor_source_sha256": hashlib.sha256(
                    executor_source.read_bytes()
                ).hexdigest(),
                "route_sources": source_files,
                "dataset_capability_sha256": context.dataset_capability[
                    "capability_sha256"
                ],
            }
        )
        native_receipt_sha256 = canonical_sha256(dict(native_identity))
        executor_receipt_sha256 = canonical_sha256(
            {
                "capability": capability,
                "hyperparameters": route_contract.get("hyperparameters"),
                "seed_identity": route_contract.get("seed_identity"),
                "source_receipt_sha256": source_receipt_sha256,
                "native_receipt_sha256": native_receipt_sha256,
                "representation_sha256_before": before,
                "representation_sha256_after": after,
            }
        )
        receipt = {
            "schema_version": 1,
            "kind": RUNTIME_SMOKE_KIND,
            "status": RUNTIME_SMOKE_STATUS,
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
            "paper_evidence_eligible": False,
            "smoke": True,
            "device": "mps",
            "optimizer_steps": 1,
            "train_render_count": 1,
            "backward_passes": 1,
            "optimizer_updates": 1,
            "heldout_render_count": 1,
            "finite_train_loss": True,
            "finite_gradients": finite_gradients,
            "parameter_update_observed": before != after,
            "finite_heldout_rgb": finite_heldout,
            "target_pixels": 1,
            "rasterized_pixels": height * width,
            "parameter_count": sum(int(parameter.numel()) for parameter in parameters),
            "parameter_bytes": sum(
                int(parameter.numel() * parameter.element_size())
                for parameter in parameters
            ),
            "sampled_peak_process_rss_bytes": _maximum_rss_bytes(),
            "sampled_peak_mps_driver_allocated_bytes": int(
                memory["sampled_peak_driver_allocated_bytes"]
            ),
            "elapsed_s": float(time.perf_counter() - started_at),
            "representation_sha256_before": before,
            "representation_sha256_after": after,
            "executor_receipt_sha256": executor_receipt_sha256,
            "native_receipt_sha256": native_receipt_sha256,
            "source_receipt_sha256": source_receipt_sha256,
        }
        return validate_public_quality_runtime_smoke(receipt, context=context)
    finally:
        if session is not None:
            session.close()


__all__ = [
    "GaussianPublicQualitySession",
    "REPRESENTATION_SEED_KEYS",
    "calibrated_camera_for_chunk",
    "camera_to_device",
    "executor_capability",
    "load_fresh_representation_seed",
    "model_state_sha256",
    "representation_seed_identity",
    "representation_seed_content_sha256",
    "run_gaussian_public_quality_runtime_smoke",
]
