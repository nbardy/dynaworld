"""Training must not receive an image with silently dropped primitives."""
import pytest
import torch

from star_uvt_runtime import ensure_star_uvt_on_path

ensure_star_uvt_on_path(include_dynaworld_root=False)
from torch_gsplat_bridge_star_uvt import UVTRenderConfig, render_uvt_tubes, render_uvt_tubes_gated


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
@pytest.mark.parametrize("gated", [False, True])
def test_overfull_tile_is_rejected_before_training_receives_rgb(monkeypatch, gated):
    monkeypatch.setenv("STAR_UVT_TILE_CAPACITY", "128")
    cfg = UVTRenderConfig(height=1, width=1, frames=1, tile_capacity=128)
    count = cfg.tile_capacity + 1
    inputs = (
        torch.tensor([[0.5, 0.5, 0.0]], device="mps").repeat(count, 1),
        torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], device="mps").repeat(count, 1),
        torch.arange(1, count + 1, dtype=torch.float32, device="mps"),
        torch.zeros((count, 3), device="mps"),
        torch.full((count,), 0.01, device="mps"),
        torch.tensor([[1.0, 0.0, 0.0]], device="mps").repeat(count, 1),
    )
    # Explicit diagnostic rendering still retains the negative result and counts.
    diagnostic = render_uvt_tubes(*inputs, cfg, return_aux=True)
    assert int(diagnostic.tile_counts.max().cpu()) == count
    assert bool(diagnostic.tile_overflow.any().cpu())
    assert abs(float(diagnostic.image[0, 0, 0, 0].cpu()) - (1.0 - 0.99**count)) > 0.001
    with pytest.raises(RuntimeError, match="tile capacity.*overflow"):
        if gated:
            render_uvt_tubes_gated(
                *inputs, torch.zeros(count, dtype=torch.int32, device="mps"),
                torch.ones(count, dtype=torch.int32, device="mps"), cfg,
            )
        else:
            render_uvt_tubes(*inputs, cfg)
