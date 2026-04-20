# Dynaworld Interface Cleanup And Unification

## Goal

Reduce script drift in `dynaworld/` without overbuilding a framework.

The problem is not that the codebase has "too many abstractions." The problem is the opposite: several training paths now share the same conceptual pipeline, but the contracts between data loading, model decode, rendering, and logging are mostly implicit and tuple-based.

The target is:

1. keep experiment velocity high
2. avoid copy-paste drift across baselines
3. make trainer/model/renderer boundaries explicit enough that new baselines fork cleanly

## Current State

### Training entrypoints

There are multiple training scripts with overlapping logic:

- `src/train/dynamicTokenGS.py`
- `src/train/train_camera_implicit_dynamic.py`
- `src/train/train_video_token_implicit_dynamic.py`

Only the video-token path currently has a real `Trainer` class. The others are still procedural script-local training loops.

### Model layer

There are multiple model families:

- known-camera image baseline
- implicit-camera image baseline
- implicit-camera video baseline

These share a lot of behavior, but not a stable forward/decode contract.

### Renderer layer

The renderer math is reasonably centralized:

- `src/train/renderers/common.py`
- `src/train/renderers/dense.py`
- `src/train/renderers/tiled.py`

But render dispatch is duplicated in each train script under slightly different helper names.

## Main Problems

### 1. Tuple soup

Models mostly return positional tuples like:

- `xyz`
- `scales`
- `quats`
- `opacities`
- `rgbs`
- `cameras`
- `camera_state`

This works for fast iteration, but it is easy to drift or reorder fields accidentally.

### 2. Implicit-camera logic is duplicated

The following concepts exist in both image and video implicit-camera models:

- zero-init camera heads
- SE(3) helper math
- camera composition
- base camera head
- path camera head
- canonical/world-space Gaussian decode

These should live in one shared implicit-camera module.

### 3. Render dispatch is duplicated

Three separate helpers do nearly the same job:

- `render_one_frame(...)`
- `render_implicit_frame(...)`
- `render_clip_frame(...)`

This should be one render dispatch helper with one config path.

### 4. Trainer structure is inconsistent

The video baseline now has clean method boundaries:

- sample
- forward
- build losses
- reconstruction backward
- logging
- validation

The older baselines still duplicate this logic inline.

### 5. Data payloads are untyped

Sequence data and sampled clips are mostly loose dicts with overlapping keys:

- `frames`
- `frame_times`
- `video_fps`
- `frame_source`
- optional `cameras`

This makes loaders flexible, but weakens downstream contracts.

## Cleanup Direction

Do not build a giant base-class hierarchy.

Instead, add a few small explicit interfaces and move shared math into the right layer.

## Proposed Contracts

The goal is **not** one shared `BaseTrainer`. The baselines have different training loops and should keep separate readable trainer classes. The cleanup target is shared typed contracts at the boundaries:

- loaders return the same sequence payload shape
- models return named gaussian/camera payloads instead of tuples
- renderers consume one stable gaussian frame type
- implicit-camera math lives in one module
- logging and validation helpers reuse the same payload vocabulary

### A. Runtime payload types

Create `src/train/runtime_types.py`.

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from camera import CameraSpec

Tensor = torch.Tensor

FrameSource = Literal[
    "camera_json",
    "summary_video",
    "explicit_video",
    "summary_sampled",
    "all_frames",
]
RendererMode = Literal["auto", "dense", "tiled"]
ResolvedRendererMode = Literal["dense", "tiled"]
ReconstructionBackwardStrategy = Literal["batched", "microbatch", "framewise"]
```

```python
@dataclass(frozen=True)
class SequenceData:
    """A full training sequence after loading and resizing.

    frames: [T, 3, H, W], float, 0..1
    frame_times: [T, 1], normalized to 0..1 unless source timestamps are unavailable
    cameras: length T when cameras are known, None for implicit-camera baselines
    """

    frames: Tensor
    frame_times: Tensor
    video_fps: float
    frame_source: FrameSource
    frame_paths: tuple[Path, ...] = ()
    cameras: tuple[CameraSpec, ...] | None = None
    records: tuple[Mapping[str, Any], ...] = ()
    intrinsics_summary: Mapping[str, float] = field(default_factory=dict)

    @property
    def frame_count(self) -> int: ...

    @property
    def image_size(self) -> int: ...

    def to(self, device: torch.device | str) -> "SequenceData": ...
```

```python
@dataclass(frozen=True)
class ClipBatch:
    """A sampled training window.

    frames: [K, 3, H, W]
    frame_times: [K, 1]
    frame_indices: [K]
    cameras: length K only for known-camera training
    """

    frames: Tensor
    frame_times: Tensor
    frame_indices: Tensor
    video_fps: float
    cameras: tuple[CameraSpec, ...] | None = None

    @property
    def frame_count(self) -> int: ...

    def as_video_batch(self) -> Tensor:
        """Return [1, K, 3, H, W] for video-token models."""
```

```python
@dataclass(frozen=True)
class CameraState:
    """Predicted camera diagnostics and regularization payload.

    fov_degrees: scalar tensor
    radius: scalar tensor
    global_residuals: [2]
    rotation_delta: [T, 3]
    translation_delta: [T, 3]
    path_residuals: [T, 6] when available
    """

    fov_degrees: Tensor
    radius: Tensor
    global_residuals: Tensor
    rotation_delta: Tensor
    translation_delta: Tensor
    path_residuals: Tensor | None = None

    def motion_features(self) -> Tensor:
        """Return [T, 6] for motion/temporal camera regularizers."""
```

```python
@dataclass(frozen=True)
class GaussianFrame:
    """One renderable gaussian frame.

    xyz: [G, 3]
    scales: [G, 3]
    quats: [G, 4], normalized
    opacities: [G, 1]
    rgbs: [G, 3], 0..1
    """

    xyz: Tensor
    scales: Tensor
    quats: Tensor
    opacities: Tensor
    rgbs: Tensor

    @property
    def gaussian_count(self) -> int: ...

    def float(self) -> "GaussianFrame": ...
```

```python
@dataclass(frozen=True)
class GaussianSequence:
    """Decoded model output for K frames.

    Tensor shapes are [K, G, C].
    cameras is present for implicit-camera outputs and known-camera render payloads.
    camera_state is present only for implicit-camera models.
    """

    xyz: Tensor
    scales: Tensor
    quats: Tensor
    opacities: Tensor
    rgbs: Tensor
    cameras: tuple[CameraSpec, ...] | None = None
    camera_state: CameraState | None = None

    @property
    def frame_count(self) -> int: ...

    @property
    def gaussian_count(self) -> int: ...

    def frame(self, index: int) -> GaussianFrame: ...
```

```python
@dataclass(frozen=True)
class StepLosses:
    total: Tensor
    reconstruction: Tensor
    camera_motion: Tensor | None = None
    camera_temporal: Tensor | None = None
    camera_global: Tensor | None = None

    def scalar_payload(self) -> dict[str, float]: ...
```

```python
@dataclass(frozen=True)
class TrainStepResult:
    batch: ClipBatch
    decoded: GaussianSequence
    losses: StepLosses
    preview_render: Tensor | None = None
```

### B. Loader signatures

Create `src/train/data_loading.py`.

```python
def load_known_camera_sequence(
    camera_json_path: Path,
    *,
    target_size: int,
    camera_image_size: int,
    max_frames: int,
    focal_mode: Literal["per_frame", "median"],
    device: torch.device,
) -> SequenceData: ...
```

```python
def load_frame_sequence(
    sequence_dir: Path,
    *,
    frames_dir: Path | None,
    video_path: Path | None,
    frame_source: FrameSource,
    target_size: int,
    max_frames: int,
    device: torch.device,
) -> SequenceData: ...
```

```python
def sample_contiguous_clip(
    sequence: SequenceData,
    *,
    frame_count: int,
    step: int | None = None,
    generator: torch.Generator | None = None,
) -> ClipBatch: ...
```

```python
def full_sequence_clip(sequence: SequenceData) -> ClipBatch: ...
```

### C. Render dispatch signatures

Create `src/train/rendering.py`.

```python
@dataclass(frozen=True)
class RenderConfig:
    renderer: RendererMode = "auto"
    auto_dense_limit: int = 400_000
    tile_size: int = 8
    bound_scale: float = 3.0
    alpha_threshold: float = 1.0 / 255.0
```

```python
def resolve_renderer_mode(
    config: RenderConfig,
    *,
    gaussian_count: int,
    height: int,
    width: int,
) -> ResolvedRendererMode: ...
```

```python
def render_gaussian_frame(
    gaussian: GaussianFrame,
    camera: CameraSpec,
    *,
    height: int,
    width: int,
    config: RenderConfig,
    dense_grid: Tensor | None = None,
) -> Tensor:
    """Return [3, H, W]."""
```

```python
def render_gaussian_sequence(
    decoded: GaussianSequence,
    cameras: Sequence[CameraSpec],
    *,
    height: int,
    width: int,
    config: RenderConfig,
    dense_grid: Tensor | None = None,
) -> Tensor:
    """Return [K, 3, H, W]."""
```

### D. Implicit-camera shared module

Create `src/train/implicit_camera.py`.

The existing duplicated definitions in `dynamic_token_gs_implicit_camera.py` and `dynamic_video_token_gs_implicit_camera.py` should move here.

```python
def build_zero_init_head(in_dim: int, out_dim: int) -> torch.nn.Sequential: ...
```

```python
def skew_symmetric(vectors: Tensor) -> Tensor:
    """vectors: [N, 3], return [N, 3, 3]."""
```

```python
def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """axis_angle: [N, 3], return [N, 3, 3]."""
```

```python
def compose_camera_with_se3_delta(
    base_camera: CameraSpec,
    rotation_delta: Tensor,
    translation_delta: Tensor,
) -> tuple[CameraSpec, ...]:
    """rotation_delta and translation_delta: [T, 3]."""
```

```python
class GlobalCameraHead(torch.nn.Module):
    def __init__(
        self,
        feat_dim: int,
        *,
        base_fov_degrees: float = 60.0,
        base_radius: float = 3.0,
        max_fov_delta_degrees: float = 15.0,
        max_radius_scale: float = 1.5,
    ) -> None: ...

    def forward(
        self,
        camera_token: Tensor,
        *,
        image_size: int,
    ) -> tuple[CameraSpec, CameraState]: ...
```

```python
class PathCameraHead(torch.nn.Module):
    def __init__(
        self,
        feat_dim: int,
        *,
        max_rotation_degrees: float = 5.0,
        max_translation_ratio: float = 0.2,
    ) -> None: ...

    def forward(
        self,
        path_tokens: Tensor,
        *,
        base_radius: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return rotation_delta [T, 3], translation_delta [T, 3], raw_residuals [T, 6]."""
```

### E. Model protocols

Create `src/train/model_protocols.py`.

These are typing contracts only. They should not force inheritance.

```python
from typing import Protocol

class KnownCameraDynamicModel(Protocol):
    def forward(
        self,
        frames: Tensor,
        *,
        cameras: Sequence[CameraSpec],
        frame_times: Tensor,
    ) -> GaussianSequence: ...
```

```python
class ImageImplicitCameraDynamicModel(Protocol):
    def forward(
        self,
        frames: Tensor,
        *,
        frame_times: Tensor,
        global_camera_token: Tensor | None = None,
    ) -> GaussianSequence: ...
```

```python
class VideoImplicitCameraDynamicModel(Protocol):
    def forward(
        self,
        clip_frames: Tensor,
        *,
        decode_times: Tensor,
    ) -> GaussianSequence:
        """clip_frames: [1, K, 3, H, W]."""
```

### F. Separate trainer shapes

Keep three trainer classes. Do not add a shared base class unless duplication remains painful after the payload cleanup.

Create or converge toward:

- `src/train/trainers/known_camera.py`
- `src/train/trainers/image_implicit_camera.py`
- `src/train/trainers/video_implicit_camera.py`

```python
@dataclass(frozen=True)
class KnownCameraTrainerConfig:
    steps: int
    lr: float
    frames_per_step: int
    eval_batch_size: int
    amp: bool
    log_every: int
    image_log_every: int
    video_log_every: int
    wandb_project: str
    wandb_run_name: str
```

```python
class KnownCameraTrainer:
    def __init__(
        self,
        *,
        model: DynamicTokenGS,
        sequence: SequenceData,
        render_config: RenderConfig,
        train_config: KnownCameraTrainerConfig,
        device: torch.device,
    ) -> None: ...

    def sample_batch(self, step: int) -> ClipBatch: ...
    def forward_batch(self, batch: ClipBatch) -> GaussianSequence: ...
    def build_losses(self, batch: ClipBatch, decoded: GaussianSequence) -> StepLosses: ...
    def train_step(self, step: int) -> TrainStepResult: ...
    def render_validation_video(self) -> Tensor: ...
    def log_step(self, step: int, result: TrainStepResult) -> None: ...
    def run(self) -> None: ...
```

```python
@dataclass(frozen=True)
class ImageImplicitCameraTrainerConfig:
    steps: int
    lr: float
    frames_per_step: int
    eval_batch_size: int
    amp: bool
    camera_motion_weight: float
    camera_temporal_weight: float
    camera_global_weight: float
    log_every: int
    image_log_every: int
    video_log_every: int
    wandb_project: str
    wandb_run_name: str
```

```python
class ImageImplicitCameraTrainer:
    def __init__(
        self,
        *,
        model: DynamicTokenGSImplicitCamera,
        sequence: SequenceData,
        render_config: RenderConfig,
        train_config: ImageImplicitCameraTrainerConfig,
        device: torch.device,
    ) -> None: ...

    def sample_batch(self, step: int) -> ClipBatch: ...
    def forward_batch(self, batch: ClipBatch) -> GaussianSequence: ...
    def build_camera_losses(self, camera_state: CameraState) -> tuple[Tensor, Tensor, Tensor]: ...
    def build_losses(self, batch: ClipBatch, decoded: GaussianSequence) -> StepLosses: ...
    def train_step(self, step: int) -> TrainStepResult: ...
    def render_validation_video(self) -> tuple[Tensor, CameraState]: ...
    def log_step(self, step: int, result: TrainStepResult) -> None: ...
    def run(self) -> None: ...
```

```python
@dataclass(frozen=True)
class VideoImplicitCameraTrainerConfig:
    steps: int
    lr: float
    train_frame_count: int
    temporal_microbatch_size: int
    recon_backward_strategy: ReconstructionBackwardStrategy
    amp: bool
    camera_motion_weight: float
    camera_temporal_weight: float
    camera_global_weight: float
    log_every: int
    image_log_every: int
    video_log_every: int
    always_log_last_step: bool
    wandb_project: str
    wandb_run_name: str
```

```python
class VideoImplicitCameraTrainer:
    def __init__(
        self,
        *,
        model: DynamicVideoTokenGSImplicitCamera,
        sequence: SequenceData,
        render_config: RenderConfig,
        train_config: VideoImplicitCameraTrainerConfig,
        device: torch.device,
    ) -> None: ...

    def sample_batch(self, step: int) -> ClipBatch: ...
    def forward_clip(self, batch: ClipBatch) -> GaussianSequence: ...
    def render_clip_frame(self, decoded: GaussianSequence, frame_index: int) -> Tensor: ...
    def build_camera_losses(self, camera_state: CameraState) -> tuple[Tensor, Tensor, Tensor]: ...
    def build_losses(self, batch: ClipBatch, decoded: GaussianSequence) -> StepLosses: ...
    def backward_reconstruction(self, batch: ClipBatch, decoded: GaussianSequence) -> Tensor: ...
    def train_step(self, step: int) -> TrainStepResult: ...
    def render_validation_video(self) -> tuple[Tensor, CameraState]: ...
    def log_step(self, step: int, result: TrainStepResult) -> None: ...
    def run(self) -> None: ...
```

### G. Logging helpers

Create `src/train/logging_utils.py`.

```python
def make_wandb_video(frames: Tensor, *, fps: float, caption: str | None = None) -> wandb.Video: ...
```

```python
def make_preview_image(gt: Tensor, pred: Tensor, *, caption: str) -> wandb.Image:
    """gt and pred: [3, H, W]."""
```

```python
def camera_metrics(camera_state: CameraState, *, prefix: str = "Camera") -> dict[str, float]: ...
```

```python
def loss_metrics(losses: StepLosses, *, prefix: str = "Loss") -> dict[str, float]: ...
```

### H. Thin script signatures

Entrypoints should shrink toward config assembly and trainer construction.

```python
def build_arg_parser() -> argparse.ArgumentParser: ...
def parse_args() -> argparse.Namespace: ...
def config_from_args(args: argparse.Namespace) -> TrainerConfig: ...
def build_trainer(config: TrainerConfig) -> KnownCameraTrainer | ImageImplicitCameraTrainer | VideoImplicitCameraTrainer: ...
def main() -> None: ...
```

## Recommended Order

### Phase 1: cheap and high value

1. add `SequenceData`, `ClipBatch`, `GaussianFrame`, and `GaussianSequence`
2. add one shared render dispatch helper around `RenderConfig`
3. extract shared implicit-camera heads and SE(3) math

### Phase 2: trainer cleanup

4. split each dynamic baseline into its own explicit trainer class
5. move common logging, validation, and camera-loss helpers into shared utilities

### Phase 3: data contract cleanup

6. make loaders return `SequenceData`
7. make models return `GaussianSequence`
8. delete compatibility shims once the new entrypoints are stable

## Non-Goals

Avoid these unless they become clearly necessary:

- a giant abstract `BaseTrainer`
- a large framework-style registry
- deep inheritance across model families
- a rewrite of the renderer math layer
- forcing known-camera, image-implicit, and video-implicit training into one loop

The cleanup should make experiments easier to fork, not harder to understand.

## Success Criteria

This cleanup is done when:

1. new baselines do not need to copy a whole train script
2. implicit-camera logic exists in one place
3. renderer dispatch exists in one place
4. model outputs are no longer positional tuple soup
5. each dynamic baseline has a separate readable trainer with the same payload vocabulary
6. known-camera code has no camera-loss branches, and implicit-camera code keeps camera-loss logic explicit
