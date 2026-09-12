"""A reduced primitive budget must retain the declared camera/time coverage."""
from types import SimpleNamespace

import pytest
import torch

from star_uvt_runtime import ensure_star_uvt_on_path

ensure_star_uvt_on_path(include_dynaworld_root=False)
from research_project.benchmarks.multicam_heldout_compare import (
    WorldTubeModel,
    initialize_world_tubes_from_train_views,
)


@pytest.mark.parametrize("tube_count,active_count", [(128, 64), (256, 128), (257, 128)])
def test_coarse_world_tubes_cover_both_cameras_and_all_times(tube_count, active_count):
    frames = torch.full((2, 32, 3, 8, 8), 0.5)
    # Red identifies the source view, green the source time, independently of XYZ.
    frames[0, :, 0] = 0.25
    frames[1, :, 0] = 0.75
    frames[:, :, 1] = torch.linspace(0.1, 0.9, 32)[None, :, None, None]
    bundle = SimpleNamespace(
        train_frames=frames,
        train_K=torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]).repeat(2, 1, 1),
        train_w2c=torch.eye(4).repeat(2, 32, 1, 1),
    )
    xyz, color, t0 = initialize_world_tubes_from_train_views(
        bundle, tube_count=tube_count, init_depth=2.0, seed=17,
        init_views="all_train", init_sampling="random", init_frames="all",
    )
    model = WorldTubeModel(
        init_x0=xyz, init_color=color, init_t0=t0, frames=32,
        init_precision_xy=30.0, init_lambda_t=0.35, init_opacity=0.35,
        min_precision_xy=1e-5, min_lambda_t=1e-5,
        velocity_reg_weight=0.0, depth_velocity_reg_weight=0.0,
        position_reg_weight=0.0,
    )
    for count in (active_count, tube_count):
        model.set_active_tube_count(count)
        batch = model.batch()
        view = (batch.color[:, 0] > 0.5).long()
        time = (batch.t0 + 15.5).long()
        group_counts = torch.bincount(view * 32 + time, minlength=64)
        assert int(group_counts.min()) == count // 64
        assert int(group_counts.max()) == (count + 63) // 64
        torch.testing.assert_close(batch.color[:, 1], torch.linspace(0.1, 0.9, 32)[time])
