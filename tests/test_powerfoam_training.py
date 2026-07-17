from __future__ import annotations

import torch

from powerfoam_training import powerfoam_train_batch_indices


def test_powerfoam_train_batch_indices_uses_frames_per_step_and_bounds() -> None:
    generator_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(7)
        indices = powerfoam_train_batch_indices(
            5,
            {"train": {"frames_per_step": "3"}},
            device=torch.device("cpu"),
        )
    finally:
        torch.random.set_rng_state(generator_state)

    assert indices.shape == (3,)
    assert indices.dtype == torch.long
    assert int(indices.min()) >= 0
    assert int(indices.max()) < 5
