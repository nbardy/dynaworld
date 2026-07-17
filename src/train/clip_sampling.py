from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from runtime_types import ClipBatch, SequenceData
from sequence_data import make_clip
from temporal_sampling import select_frame_indices


def sample_clip_batch(
    sequence: SequenceData,
    *,
    train_frame_count: int,
    frame_sampling: Mapping[str, Any],
    device: torch.device | str,
) -> ClipBatch:
    """Sample a typed clip window from one sequence.

    This is the shared boundary for trainer sampling: temporal policy chooses
    frame indices, then `sequence_data.make_clip` slices frames/times/cameras
    into a `ClipBatch`. Trainers may still adapt the batch to legacy
    `(clip_frames, clip_times)` tensors at their own boundary.
    """

    frame_indices = select_frame_indices(
        sequence.frame_count,
        int(train_frame_count),
        frame_sampling,
        device=device,
    )
    return make_clip(sequence, frame_indices)


__all__ = ["sample_clip_batch"]
