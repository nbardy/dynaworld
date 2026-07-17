from __future__ import annotations

from powerfoam_eval_render import powerfoam_eval_batch_size


def test_powerfoam_eval_batch_size_uses_positive_frames_per_step() -> None:
    assert powerfoam_eval_batch_size({"train": {"frames_per_step": "3"}}) == 3
    assert powerfoam_eval_batch_size({"train": {"frames_per_step": 0}}) == 1
