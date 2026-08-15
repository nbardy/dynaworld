from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from paper_training_types import (
    ImageSize,
    PaperCostSnapshot,
    PaperDatasetContract,
    PaperStage,
    PaperTrainingProtocol,
    SpacetimeBatch,
    SpacetimeSample,
)

if TYPE_CHECKING:
    import torch

PAPER_SAMPLE_SCHEDULE_SCHEMA_VERSION = 1
PAPER_SAMPLE_SCHEDULE_ALGORITHM = "spacetime_epoch_v1"
PAPER_DATASET_BUNDLE_SCHEMA_VERSION = 1
PAPER_EVALUATOR_SCHEMA_VERSION = 2
PAPER_RUNTIME_SCHEMA_VERSION = 1
PAPER_RUNTIME_SOURCE_TREE_SCHEMA_VERSION = 1
LPIPS_ALEXNET_TRUNK = {
    "filename": "alexnet-owt-7be5be79.pth",
    "bytes": 244_408_911,
    "sha256": "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
}
LPIPS_ALEX_V01_LINEAR = {
    "resource": "weights/v0.1/alex.pth",
    "bytes": 6_009,
    "sha256": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
}


def synchronize_device(device: torch.device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def serialized_state_dict_bytes(model: torch.nn.Module) -> int:
    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


class PaperPhaseTimer:
    """Device-synchronized cold-forward, steady-forward, backward, and optimizer timing."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.totals = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0}
        self.counts = {"forward": 0, "backward": 0, "optimizer": 0}
        self.cold_compile_forward_s: float | None = None

    @contextmanager
    def measure(self, phase: str):
        started_at = self.start(phase)
        yield
        self.stop(phase, started_at)

    def start(self, phase: str) -> float:
        if phase not in self.totals:
            raise ValueError(f"unsupported paper timing phase: {phase}")
        synchronize_device(self.device)
        return time.perf_counter()

    def stop(self, phase: str, started_at: float) -> float:
        synchronize_device(self.device)
        elapsed = time.perf_counter() - started_at
        if phase == "forward" and self.cold_compile_forward_s is None:
            self.cold_compile_forward_s = elapsed
        else:
            self.totals[phase] += elapsed
            self.counts[phase] += 1
        return elapsed

    def snapshot(self, *, train_wall_s: float) -> dict[str, Any]:
        cold = 0.0 if self.cold_compile_forward_s is None else self.cold_compile_forward_s
        return {
            "definition": "device-synchronized; cold_compile_forward is the first forward including lazy kernel compilation",
            "cold_compile_forward_s": cold,
            "steady_forward_s": self.totals["forward"],
            "steady_forward_calls": self.counts["forward"],
            "backward_s": self.totals["backward"],
            "backward_calls": self.counts["backward"],
            "optimizer_s": self.totals["optimizer"],
            "optimizer_calls": self.counts["optimizer"],
            "train_wall_s": float(train_wall_s),
            "steady_forward_mean_s": (
                self.totals["forward"] / self.counts["forward"] if self.counts["forward"] else 0.0
            ),
            "backward_mean_s": (
                self.totals["backward"] / self.counts["backward"] if self.counts["backward"] else 0.0
            ),
            "optimizer_mean_s": (
                self.totals["optimizer"] / self.counts["optimizer"] if self.counts["optimizer"] else 0.0
            ),
        }


def normalize_image_size(value: Any, *, name: str = "image size") -> ImageSize:
    if isinstance(value, ImageSize):
        return value
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, [height, width], or object")
    if isinstance(value, int):
        return ImageSize(value, value)
    if isinstance(value, Mapping):
        return ImageSize(int(value["height"]), int(value["width"]))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return ImageSize(int(value[0]), int(value[1]))
    raise ValueError(f"{name} must be an integer, [height, width], or object")


def normalize_paper_stages(
    raw_stages: Any,
    *,
    total_steps: int,
    default_image_size: ImageSize,
    default_primitive_count: int,
    default_frames_per_step: int,
) -> tuple[PaperStage, ...]:
    if int(total_steps) < 1:
        raise ValueError("total_steps must be positive")
    if raw_stages is None:
        return (
            PaperStage(
                label="fixed",
                start_step=0,
                end_step=int(total_steps),
                image_size=default_image_size,
                primitive_count=int(default_primitive_count),
                frames_per_step=int(default_frames_per_step),
            ),
        )
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("paper_protocol.stages must be a non-empty list or null")

    stages: list[PaperStage] = []
    start_step = 0
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, Mapping):
            raise ValueError("paper_protocol.stages entries must be objects")
        end_step = int(raw["until_step"])
        raw_image_size = raw.get("image_size")
        if raw_image_size is None:
            if raw.get("height") is None and raw.get("width") is None:
                raw_image_size = default_image_size
            elif raw.get("height") is not None and raw.get("width") is not None:
                raw_image_size = {"height": raw["height"], "width": raw["width"]}
            else:
                raise ValueError(
                    f"paper_protocol.stages[{index}] must provide both height and width"
                )
        image_size = normalize_image_size(
            raw_image_size,
            name=f"paper_protocol.stages[{index}].image_size",
        )
        stages.append(
            PaperStage(
                label=str(raw.get("label", f"stage_{index}")),
                start_step=start_step,
                end_step=end_step,
                image_size=image_size,
                primitive_count=int(raw.get("primitive_count", default_primitive_count)),
                frames_per_step=int(raw.get("frames_per_step", default_frames_per_step)),
                lr_multiplier=float(raw.get("lr_multiplier", 1.0)),
            )
        )
        start_step = end_step

    if stages[-1].end_step != int(total_steps):
        raise ValueError("the final paper stage until_step must equal the training step count")
    for previous, current in zip(stages, stages[1:]):
        if current.start_step != previous.end_step:
            raise ValueError("paper stages must be contiguous")
        if current.image_size.height < previous.image_size.height or current.image_size.width < previous.image_size.width:
            raise ValueError("paper stage image sizes must be non-decreasing")
        if current.primitive_count < previous.primitive_count:
            raise ValueError("paper stage primitive counts must be non-decreasing")
    return tuple(stages)


def paper_stage_for_step(stages: tuple[PaperStage, ...], step: int) -> PaperStage:
    for stage in stages:
        if stage.contains(step):
            return stage
    raise IndexError(f"step {step} is outside the paper stage schedule")


def resolve_paper_training_protocol(raw: Mapping[str, Any]) -> PaperTrainingProtocol:
    if not bool(raw.get("enabled", False)):
        raise ValueError("paper protocol requires enabled=true")
    dataset_raw = raw.get("dataset")
    if not isinstance(dataset_raw, Mapping):
        raise ValueError("paper protocol dataset must be an object")
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("paper protocol stages must be a non-empty list")
    final_raw = raw_stages[-1]
    if not isinstance(final_raw, Mapping):
        raise ValueError("paper protocol stage entries must be objects")
    final_image_size = normalize_image_size(final_raw.get("image_size"), name="final paper image size")
    final_primitive_count = int(final_raw["primitive_count"])
    default_frames_per_step = int(raw.get("frames_per_step", final_raw.get("frames_per_step", 1)))
    steps = int(raw["steps"])
    stages = normalize_paper_stages(
        raw_stages,
        total_steps=steps,
        default_image_size=final_image_size,
        default_primitive_count=final_primitive_count,
        default_frames_per_step=default_frames_per_step,
    )
    dataset = PaperDatasetContract(
        manifest=str(dataset_raw["manifest"]),
        sample_id=str(dataset_raw["sample_id"]),
        train_cameras=tuple(str(value) for value in dataset_raw["train_cameras"]),
        heldout_cameras=tuple(str(value) for value in dataset_raw["heldout_cameras"]),
        frame_count=int(dataset_raw["frame_count"]),
        fps=float(dataset_raw["fps"]),
    )
    return PaperTrainingProtocol(
        name=str(raw["name"]),
        dataset=dataset,
        steps=steps,
        max_train_seconds=float(raw["max_train_seconds"]),
        same_time_count=int(raw.get("same_time_count", 1)),
        local_time_count=int(raw.get("local_time_count", 0)),
        local_time_radius=int(raw.get("local_time_radius", 0)),
        sampler_seed_offset=int(raw.get("sampler_seed_offset", 7001)),
        stages=stages,
    )


def apply_paper_dataset_contract(
    data_cfg: Mapping[str, Any],
    protocol: Mapping[str, Any] | PaperTrainingProtocol | None,
) -> dict[str, Any]:
    resolved = dict(data_cfg)
    if protocol is None:
        return resolved
    paper = protocol if isinstance(protocol, PaperTrainingProtocol) else resolve_paper_training_protocol(protocol)
    if len(paper.dataset.heldout_cameras) != 1:
        raise ValueError("the current multicam loader requires exactly one heldout camera")
    resolved.update(
        {
            "frame_source": "multicam_val",
            "max_frames": paper.dataset.frame_count,
            "multicam_manifest": paper.dataset.manifest,
            "multicam_sample_id": paper.dataset.sample_id,
            "multicam_train_cameras": list(paper.dataset.train_cameras),
            "multicam_heldout_camera": paper.dataset.heldout_cameras[0],
            "multicam_anchor_camera": paper.dataset.train_cameras[0],
        }
    )
    return resolved


class SpacetimeEpochSampler:
    """Coverage-exact shuffled epochs with best-effort spatial/temporal grouping."""

    def __init__(
        self,
        *,
        view_count: int,
        frame_indices: Sequence[int],
        batch_size: int,
        same_time_count: int,
        local_time_count: int,
        local_time_radius: int,
        seed: int,
    ) -> None:
        if int(view_count) < 1:
            raise ValueError("view_count must be positive")
        frames = tuple(int(frame) for frame in frame_indices)
        if not frames or len(set(frames)) != len(frames):
            raise ValueError("frame_indices must be non-empty and unique")
        if min(frames) < 0:
            raise ValueError("frame_indices must be non-negative")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if int(same_time_count) < 1:
            raise ValueError("same_time_count must be at least one because it includes the anchor")
        if int(local_time_count) < 0 or int(local_time_radius) < 0:
            raise ValueError("local_time_count and local_time_radius must be non-negative")
        if int(same_time_count) + int(local_time_count) > int(batch_size):
            raise ValueError("same_time_count + local_time_count must not exceed batch_size")
        self.view_count = int(view_count)
        self.frame_indices = frames
        self.batch_size = int(batch_size)
        self.same_time_count = int(same_time_count)
        self.local_time_count = int(local_time_count)
        self.local_time_radius = int(local_time_radius)
        self.seed = int(seed)
        self.epoch = -1
        self.batch_index = 0
        self._remaining: list[SpacetimeSample] = []
        self._start_epoch()

    @property
    def samples_per_epoch(self) -> int:
        return self.view_count * len(self.frame_indices)

    def _start_epoch(self) -> None:
        self.epoch += 1
        self.batch_index = 0
        self._remaining = [
            SpacetimeSample(view_index=view, frame_index=frame)
            for view in range(self.view_count)
            for frame in self.frame_indices
        ]
        random.Random(self.seed + self.epoch).shuffle(self._remaining)

    def _take_matching(self, selected: list[SpacetimeSample], predicate: Any, count: int) -> None:
        if count <= 0:
            return
        matches = [sample for sample in self._remaining if predicate(sample)][:count]
        for sample in matches:
            self._remaining.remove(sample)
            selected.append(sample)

    def next_batch(self, batch_size: int | None = None) -> SpacetimeBatch:
        resolved_batch_size = self.batch_size if batch_size is None else int(batch_size)
        if resolved_batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self._remaining:
            self._start_epoch()
        epoch = self.epoch
        batch_index = self.batch_index
        anchor = self._remaining.pop(0)
        selected = [anchor]
        self._take_matching(
            selected,
            lambda sample: sample.frame_index == anchor.frame_index and sample.view_index != anchor.view_index,
            min(self.same_time_count - 1, resolved_batch_size - len(selected)),
        )
        self._take_matching(
            selected,
            lambda sample: (
                sample.view_index == anchor.view_index
                and 0 < abs(sample.frame_index - anchor.frame_index) <= self.local_time_radius
            ),
            min(self.local_time_count, resolved_batch_size - len(selected)),
        )
        fill_count = min(resolved_batch_size - len(selected), len(self._remaining))
        selected.extend(self._remaining[:fill_count])
        del self._remaining[:fill_count]
        self.batch_index += 1
        return SpacetimeBatch(
            samples=tuple(selected),
            epoch=epoch,
            batch_index=batch_index,
            completes_epoch=not self._remaining,
        )


class PaperSampleScheduleDigest:
    """Canonical digest proving that paper lanes consumed the same samples."""

    def __init__(self, *, sampler_seed: int) -> None:
        self.sampler_seed = int(sampler_seed)
        self.record_count = 0
        self._digest = hashlib.sha256()
        self._update(
            {
                "schema_version": PAPER_SAMPLE_SCHEDULE_SCHEMA_VERSION,
                "algorithm": PAPER_SAMPLE_SCHEDULE_ALGORITHM,
                "sampler_seed": self.sampler_seed,
            }
        )

    def _update(self, value: Mapping[str, Any]) -> None:
        self._digest.update(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        self._digest.update(b"\n")

    def record(
        self,
        *,
        step: int,
        stage: PaperStage,
        batch: SpacetimeBatch,
    ) -> None:
        resolved_step = int(step)
        if resolved_step != self.record_count:
            raise ValueError(
                "paper sample schedule steps must be contiguous and zero-based: "
                f"expected {self.record_count}, got {resolved_step}"
            )
        if not stage.contains(resolved_step):
            raise ValueError(
                f"paper sample schedule step {resolved_step} is outside stage {stage.label}"
            )
        if len(batch.samples) != stage.frames_per_step:
            raise ValueError(
                "paper sample schedule batch size does not match the active stage: "
                f"expected {stage.frames_per_step}, got {len(batch.samples)}"
            )
        self._update(
            {
                "step": resolved_step,
                "stage": stage.as_dict(),
                "batch": batch.as_dict(),
            }
        )
        self.record_count += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": PAPER_SAMPLE_SCHEDULE_SCHEMA_VERSION,
            "algorithm": PAPER_SAMPLE_SCHEDULE_ALGORITHM,
            "sampler_seed": self.sampler_seed,
            "record_count": self.record_count,
            "sha256": self._digest.hexdigest(),
        }


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lpips_alex_asset_status(
    *,
    torch_home: str | Path | None = None,
    lpips_package_root: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect the exact LPIPS/AlexNet assets without importing Torch or LPIPS."""

    if torch_home is None:
        configured_torch_home = os.environ.get("TORCH_HOME")
        if configured_torch_home:
            resolved_torch_home = Path(configured_torch_home).expanduser()
        else:
            cache_home = Path(
                os.environ.get(
                    "XDG_CACHE_HOME",
                    str(Path.home() / ".cache"),
                )
            ).expanduser()
            resolved_torch_home = cache_home / "torch"
    else:
        resolved_torch_home = Path(torch_home).expanduser()
    trunk_path = (
        resolved_torch_home
        / "hub"
        / "checkpoints"
        / str(LPIPS_ALEXNET_TRUNK["filename"])
    ).resolve()

    package_version = None
    try:
        package_version = importlib.metadata.version("lpips")
    except importlib.metadata.PackageNotFoundError:
        pass
    if lpips_package_root is None:
        spec = importlib.util.find_spec("lpips")
        package_root = (
            None
            if spec is None or spec.origin is None
            else Path(spec.origin).resolve().parent
        )
    else:
        package_root = Path(lpips_package_root).expanduser().resolve()
    linear_path = (
        None
        if package_root is None
        else package_root / str(LPIPS_ALEX_V01_LINEAR["resource"])
    )

    def inspect(path: Path | None, expected: Mapping[str, Any]) -> dict[str, Any]:
        exists = path is not None and path.is_file()
        size = None if not exists else int(path.stat().st_size)
        sha256 = None if not exists else _file_sha256(path)
        return {
            "path": None if path is None else str(path),
            "exists": exists,
            "bytes": size,
            "sha256": sha256,
            "expected_bytes": int(expected["bytes"]),
            "expected_sha256": str(expected["sha256"]),
            "accepted": (
                exists
                and size == int(expected["bytes"])
                and sha256 == str(expected["sha256"])
            ),
        }

    assets = {
        "alexnet_trunk": inspect(trunk_path, LPIPS_ALEXNET_TRUNK),
        "lpips_v0_1_alex_linear": inspect(
            linear_path,
            LPIPS_ALEX_V01_LINEAR,
        ),
    }
    checks = {
        "lpips_package_installed": package_version is not None,
        "alexnet_trunk_exact": assets["alexnet_trunk"]["accepted"],
        "lpips_v0_1_alex_linear_exact": assets[
            "lpips_v0_1_alex_linear"
        ]["accepted"],
    }
    return {
        "status": "pass" if all(checks.values()) else "rejected",
        "execution": "none",
        "network_download_allowed": False,
        "lpips_package_version": package_version,
        "checks": checks,
        "assets": assets,
    }


def require_lpips_alex_assets(status: Mapping[str, Any]) -> None:
    checks = status.get("checks")
    if (
        status.get("status") != "pass"
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        failed = (
            []
            if not isinstance(checks, Mapping)
            else [key for key, value in checks.items() if value is not True]
        )
        raise RuntimeError(
            "paper LPIPS/AlexNet assets are missing or drifted; prefetch the "
            "exact checked weights before execution: "
            + ", ".join(failed or ["invalid_asset_status"])
        )


def tensor_content_identity(
    value: torch.Tensor | None,
    *,
    chunk_elements: int = 1 << 20,
) -> dict[str, Any] | None:
    """Hash a contiguous tensor while staging at most one bounded chunk."""

    import torch

    if value is None:
        return None
    if int(chunk_elements) < 1:
        raise ValueError("tensor identity chunk_elements must be positive")
    if value.layout != torch.strided or not value.is_contiguous():
        raise ValueError(
            "tensor identity requires a contiguous strided tensor; callers "
            "must make layout conversion explicit before hashing"
        )
    tensor = value.detach()
    metadata = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "bytes": int(tensor.numel() * tensor.element_size()),
        "byte_order": f"native_{sys.byteorder}_endian",
        "layout": "contiguous_c_order",
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\n")
    flattened = tensor.reshape(-1)
    for start in range(0, int(flattened.numel()), int(chunk_elements)):
        chunk = flattened[start : start + int(chunk_elements)]
        if chunk.device.type != "cpu":
            chunk = chunk.to(device="cpu")
        array = chunk.contiguous().numpy()
        digest.update(memoryview(array).cast("B"))
    return {**metadata, "sha256": digest.hexdigest()}


def paper_dataset_bundle_identity(
    bundle: Any,
    *,
    image_size: ImageSize,
    decoded_frame_identities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Identity of the decoded targets and calibrated cameras consumed by a lane.

    A streaming target source may supply identities computed in bounded chunks;
    their format and digest are identical to :func:`tensor_content_identity`.
    """

    condition = bundle.condition_sequence
    if decoded_frame_identities is None:
        train_frame_identity = tensor_content_identity(bundle.train_frames)
        heldout_frame_identity = tensor_content_identity(bundle.heldout_frames)
    else:
        if "train_frames" not in decoded_frame_identities:
            raise ValueError("decoded frame identities require train_frames")
        train_frame_identity = decoded_frame_identities["train_frames"]
        heldout_frame_identity = decoded_frame_identities.get("heldout_frames")
    payload = {
        "schema_version": PAPER_DATASET_BUNDLE_SCHEMA_VERSION,
        "loader": "load_multicam_video_bundle_v1",
        "sample_id": (
            None
            if bundle.metadata is None
            else bundle.metadata.get("sample_id")
        ),
        "image_size": image_size.as_list(),
        "target_decode": {
            "dtype": "torch.float32",
            "range": [0.0, 1.0],
            "layout": "view_time_channel_height_width",
        },
        "train_camera_names": list(bundle.train_camera_names),
        "heldout_camera_names": list(bundle.heldout_camera_names or ()),
        "train_lens_models": list(bundle.train_lens_models or ()),
        "heldout_lens_models": list(bundle.heldout_lens_models or ()),
        "pose_source": bundle.pose_source,
        "frame_count": int(bundle.frame_count),
        "video_fps": float(condition.video_fps),
        "frame_times": tensor_content_identity(condition.frame_times),
        "train_frames": train_frame_identity,
        "train_K": tensor_content_identity(bundle.train_K),
        "train_w2c": tensor_content_identity(bundle.train_w2c),
        "train_distortions": tensor_content_identity(bundle.train_distortions),
        "heldout_frames": heldout_frame_identity,
        "heldout_K": tensor_content_identity(bundle.heldout_K),
        "heldout_w2c": tensor_content_identity(bundle.heldout_w2c),
        "heldout_distortions": tensor_content_identity(
            bundle.heldout_distortions
        ),
        "anchor_c2w": tensor_content_identity(bundle.anchor_c2w),
    }
    return {**payload, "sha256": _canonical_json_sha256(payload)}


def paper_evaluator_contract(
    *,
    ssim_window_size: int = 11,
    ssim_c1: float = 0.0001,
    ssim_c2: float = 0.0009,
    background: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Canonical metric semantics shared by all publication ablation lanes."""

    payload = {
        "schema_version": PAPER_EVALUATOR_SCHEMA_VERSION,
        "id": "paper_rgb_fullset_v2",
        "prediction_target_domain": (
            "decoded_rgb_float32_0_1_no_transfer_conversion"
        ),
        "prediction_clamp": "clamp_0_1",
        "color_calibration": "none",
        "fixed_background_rgb": [float(value) for value in background],
        "l1": "mean_abs_error_over_all_rgb_elements",
        "mse": "mean_squared_error_over_all_rgb_elements",
        "psnr": "-10_log10(max(mse,1e-12)); data_range=1",
        "ssim": {
            "implementation": "src/train/losses.py:ssim_per_image",
            "aggregation": "mean_over_images",
            "window_size": int(ssim_window_size),
            "small_image_window": "min(window,height,width), then previous odd, minimum 1",
            "c1": float(ssim_c1),
            "c2": float(ssim_c2),
        },
        "lpips": {
            "implementation": "src/train/perceptual_metrics.py:video_lpips",
            "network": "alex",
            "alexnet_trunk": dict(LPIPS_ALEXNET_TRUNK),
            "linear_weights": dict(LPIPS_ALEX_V01_LINEAR),
            "input": "clamp_0_1_then_map_to_minus1_plus1",
            "aggregation": "mean_over_images",
        },
        "evaluation_set": "all_declared_train_and_heldout_frames",
        "media_subsampling_does_not_change_metrics": True,
    }
    if len(payload["fixed_background_rgb"]) != 3:
        raise ValueError("paper evaluator background must contain three values")
    if int(ssim_window_size) < 1 or int(ssim_window_size) % 2 == 0:
        raise ValueError("paper evaluator SSIM window must be positive and odd")
    return {**payload, "sha256": _canonical_json_sha256(payload)}


def paper_runtime_identity() -> dict[str, Any]:
    """Stable host/runtime identity required for comparable timing evidence."""

    import torch

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def sysctl(name: str) -> str | None:
        if sys.platform != "darwin":
            return None
        try:
            return subprocess.check_output(
                ("sysctl", "-n", name),
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    lpips_assets = lpips_alex_asset_status()
    require_lpips_alex_assets(lpips_assets)
    payload = {
        "schema_version": PAPER_RUNTIME_SCHEMA_VERSION,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "build": (
                subprocess.check_output(
                    ("sw_vers", "-buildVersion"),
                    text=True,
                ).strip()
                if sys.platform == "darwin"
                else None
            ),
            "machine": platform.machine(),
        },
        "hardware": {
            "model": sysctl("hw.model"),
            "chip": sysctl("machdep.cpu.brand_string"),
            "logical_cpu_count": int(os.cpu_count() or 1),
            "physical_cpu_count": sysctl("hw.physicalcpu"),
            "physical_memory_bytes": sysctl("hw.memsize"),
        },
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "torch": {
            "version": str(torch.__version__),
            "mps_built": bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
            ),
            "mps_available": bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "packages": {
            "lpips": package_version("lpips"),
            "opencv_python": package_version("opencv-python"),
            "pillow": package_version("pillow"),
            "numpy": package_version("numpy"),
        },
        "lpips_assets": lpips_assets,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "PYTORCH_MPS_ALLOCATOR_POLICY",
            )
        },
    }
    return {**payload, "sha256": _canonical_json_sha256(payload)}


def paper_runtime_source_tree_identity(root: str | Path) -> dict[str, Any]:
    source_root = Path(root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"paper runtime source tree is missing: {source_root}"
        )
    files = []
    for path in sorted(candidate for candidate in source_root.rglob("*") if candidate.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(source_root)),
                "bytes": int(path.stat().st_size),
                "sha256": digest.hexdigest(),
            }
        )
    if not files:
        raise ValueError(f"paper runtime source tree is empty: {source_root}")
    payload = {
        "schema_version": PAPER_RUNTIME_SOURCE_TREE_SCHEMA_VERSION,
        "root": str(source_root),
        "file_count": len(files),
        "files": files,
    }
    return {**payload, "sha256": _canonical_json_sha256(payload)}


def validate_paper_runtime_source_tree_identity(
    identity: Mapping[str, Any],
) -> None:
    """Fail if any recorded runtime source file was added, removed, or changed."""

    if not isinstance(identity, Mapping):
        raise ValueError("paper runtime source tree identity must be an object")
    root = identity.get("root")
    if not isinstance(root, str) or not root:
        raise ValueError("paper runtime source tree root is invalid")
    current = paper_runtime_source_tree_identity(root)
    if dict(identity) != current:
        raise ValueError(
            "paper runtime source tree provenance drifted: the current "
            "exhaustive file set or file contents differ from the record"
        )


def paper_native_module_identity(
    module_name: str,
    *,
    runtime_source_root: str | Path,
) -> dict[str, Any]:
    module = importlib.import_module(str(module_name))
    path = Path(str(module.__file__)).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"loaded paper native module is missing: {module_name} at {path}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "module": str(module_name),
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
        "runtime_source_tree": paper_runtime_source_tree_identity(
            runtime_source_root
        ),
    }


@dataclass
class PaperRGBMetricAccumulator:
    """Global paper RGB metrics, invariant to view/chunk traversal."""

    ssim_window_size: int = 11
    ssim_c1: float = 0.0001
    ssim_c2: float = 0.0009
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    element_count: int = 0
    ssim_sum: float = 0.0
    image_count: int = 0

    @staticmethod
    def _nchw(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(
                f"paper RGB metrics require a rank-four tensor, got {tuple(value.shape)}"
            )
        if int(value.shape[1]) == 3:
            return value.float()
        if int(value.shape[-1]) == 3:
            return value.permute(0, 3, 1, 2).contiguous().float()
        raise ValueError(
            "paper RGB metrics require NCHW or NHWC tensors with three channels"
        )

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        from losses import ssim_per_image

        prediction_nchw = self._nchw(prediction).clamp(0.0, 1.0)
        target_nchw = self._nchw(target)
        if prediction_nchw.shape != target_nchw.shape:
            raise ValueError(
                "paper RGB metrics require matching prediction and target shapes, "
                f"got {tuple(prediction_nchw.shape)} and {tuple(target_nchw.shape)}"
            )
        # Promote before subtraction.  Casting an already-computed float32
        # delta preserves float32 rounding and makes the aggregate depend on
        # how views were chunked.  The paper contract requires one global
        # metric independent of traversal/layout, so accumulate the sufficient
        # statistics in float64 from the original float32 samples.
        delta = prediction_nchw.double() - target_nchw.double()
        self.absolute_error_sum += float(delta.abs().sum().item())
        self.squared_error_sum += float(delta.square().sum().item())
        self.element_count += int(delta.numel())
        window_size = min(
            int(self.ssim_window_size),
            int(prediction_nchw.shape[-2]),
            int(prediction_nchw.shape[-1]),
        )
        if window_size % 2 == 0:
            window_size -= 1
        window_size = max(window_size, 1)
        values = ssim_per_image(
            prediction_nchw,
            target_nchw,
            window_size=window_size,
            c1=float(self.ssim_c1),
            c2=float(self.ssim_c2),
        )
        self.ssim_sum += float(values.sum().item())
        self.image_count += int(values.numel())

    def metrics(self, *, prefix: str = "eval") -> dict[str, float]:
        if self.element_count < 1 or self.image_count < 1:
            raise ValueError("paper RGB metrics require at least one image")
        l1 = self.absolute_error_sum / float(self.element_count)
        mse = self.squared_error_sum / float(self.element_count)
        return {
            f"{prefix}_l1": l1,
            f"{prefix}_mse": mse,
            f"{prefix}_psnr": -10.0 * math.log10(max(mse, 1.0e-12)),
            f"{prefix}_ssim": self.ssim_sum / float(self.image_count),
        }


def resize_video_frames(frames: torch.Tensor, image_size: ImageSize) -> torch.Tensor:
    from torch.nn import functional as F

    if frames.ndim not in {4, 5}:
        raise ValueError(f"expected frames [T,C,H,W] or [V,T,C,H,W], got {tuple(frames.shape)}")
    if tuple(frames.shape[-2:]) == (image_size.height, image_size.width):
        return frames
    leading = frames.shape[:-3]
    flattened = frames.reshape(-1, *frames.shape[-3:])
    resized = F.interpolate(
        flattened,
        size=(image_size.height, image_size.width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape(*leading, *resized.shape[-3:]).contiguous()


def resize_ray_grids(rays: torch.Tensor, image_size: ImageSize) -> torch.Tensor:
    import torch
    from torch.nn import functional as F

    if rays.ndim != 4 or rays.shape[-1] != 6:
        raise ValueError(f"expected rays [B,H,W,6], got {tuple(rays.shape)}")
    if tuple(rays.shape[1:3]) == (image_size.height, image_size.width):
        return rays
    channels = rays.permute(0, 3, 1, 2)
    resized = F.interpolate(
        channels,
        size=(image_size.height, image_size.width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).contiguous()
    origins = resized[..., :3]
    directions = F.normalize(resized[..., 3:], dim=-1)
    return torch.cat((origins, directions), dim=-1).contiguous()


def scale_intrinsics(K: torch.Tensor, *, source: ImageSize, target: ImageSize) -> torch.Tensor:
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"expected intrinsics [...,3,3], got {tuple(K.shape)}")
    if source == target:
        return K
    scaled = K.clone()
    sx = float(target.width) / float(source.width)
    sy = float(target.height) / float(source.height)
    scaled[..., 0, 0] *= sx
    scaled[..., 0, 2] *= sx
    scaled[..., 1, 1] *= sy
    scaled[..., 1, 2] *= sy
    return scaled


def tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    import torch

    return sum(
        tensor_bytes(value)
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


class PaperCostTracker:
    def __init__(self) -> None:
        self.optimizer_steps = 0
        self.target_frames = 0
        self.rasterized_frames = 0
        self.target_pixels = 0
        self.rasterized_pixels = 0

    def record(self, *, stage: PaperStage, target_frames: int, rasterized_frames: int) -> None:
        self.optimizer_steps += 1
        self.target_frames += int(target_frames)
        self.rasterized_frames += int(rasterized_frames)
        self.target_pixels += int(target_frames) * stage.image_size.pixels
        self.rasterized_pixels += int(rasterized_frames) * stage.image_size.pixels

    def snapshot(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        elapsed_s: float,
        memory: Mapping[str, int] | None = None,
        serialized_checkpoint_bytes: int | None = None,
    ) -> PaperCostSnapshot:
        parameters = tuple(model.parameters())
        memory_values = memory or {}
        return PaperCostSnapshot(
            optimizer_steps=self.optimizer_steps,
            target_frames=self.target_frames,
            rasterized_frames=self.rasterized_frames,
            target_pixels=self.target_pixels,
            rasterized_pixels=self.rasterized_pixels,
            parameter_count=sum(parameter.numel() for parameter in parameters),
            trainable_parameter_count=sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            parameter_bytes=sum(tensor_bytes(parameter) for parameter in parameters),
            optimizer_state_bytes=optimizer_state_bytes(optimizer),
            serialized_checkpoint_bytes=(
                serialized_state_dict_bytes(model)
                if serialized_checkpoint_bytes is None
                else int(serialized_checkpoint_bytes)
            ),
            sampled_peak_current_allocated_bytes=int(
                memory_values.get("sampled_peak_current_allocated_bytes", 0)
            ),
            sampled_peak_driver_allocated_bytes=int(
                memory_values.get("sampled_peak_driver_allocated_bytes", 0)
            ),
            elapsed_s=float(elapsed_s),
        )


__all__ = [
    "PAPER_DATASET_BUNDLE_SCHEMA_VERSION",
    "PAPER_EVALUATOR_SCHEMA_VERSION",
    "PAPER_RUNTIME_SCHEMA_VERSION",
    "PAPER_RUNTIME_SOURCE_TREE_SCHEMA_VERSION",
    "PAPER_SAMPLE_SCHEDULE_ALGORITHM",
    "PAPER_SAMPLE_SCHEDULE_SCHEMA_VERSION",
    "LPIPS_ALEXNET_TRUNK",
    "LPIPS_ALEX_V01_LINEAR",
    "PaperCostTracker",
    "PaperPhaseTimer",
    "PaperRGBMetricAccumulator",
    "PaperSampleScheduleDigest",
    "SpacetimeEpochSampler",
    "apply_paper_dataset_contract",
    "normalize_image_size",
    "normalize_paper_stages",
    "optimizer_state_bytes",
    "lpips_alex_asset_status",
    "paper_dataset_bundle_identity",
    "paper_evaluator_contract",
    "paper_native_module_identity",
    "paper_runtime_identity",
    "paper_runtime_source_tree_identity",
    "paper_stage_for_step",
    "resolve_paper_training_protocol",
    "resize_ray_grids",
    "resize_video_frames",
    "require_lpips_alex_assets",
    "scale_intrinsics",
    "serialized_state_dict_bytes",
    "synchronize_device",
    "tensor_bytes",
    "tensor_content_identity",
    "validate_paper_runtime_source_tree_identity",
]
