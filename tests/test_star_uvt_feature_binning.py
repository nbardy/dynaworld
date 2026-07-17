from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

STAR_UVT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
if str(STAR_UVT_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_UVT_ROOT))

from torch_gsplat_bridge_star_uvt import UVTRenderConfig  # noqa: E402
from torch_gsplat_bridge_star_uvt.feature_rasterize import (  # noqa: E402
    chunked_uvt_config,
    render_uvt_feature_sparse_pixels_with_bins,
    shift_ma_for_frame_chunk,
)


def test_sparse_feature_binner_keeps_chunk_shifted_moving_tube() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the STAR UVT Metal sparse binner")

    device = torch.device("mps")
    global_config = UVTRenderConfig(height=32, width=32, frames=64)
    local_config = chunked_uvt_config(global_config, chunk_frames=2)
    ma = torch.tensor([[5.0, 8.5, 0.0]], dtype=torch.float32, device=device)
    velocity_u = 1.0
    velocity_v = 0.0
    spatial_precision = 1.0 / (40.0 * 40.0)
    temporal_precision = 1.0 / (64.0 * 64.0)
    q_uvt = torch.tensor(
        [
            [
                spatial_precision,
                0.0,
                -spatial_precision * velocity_u,
                spatial_precision,
                -spatial_precision * velocity_v,
                temporal_precision
                + spatial_precision * velocity_u * velocity_u
                + spatial_precision * velocity_v * velocity_v,
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    depth0 = torch.ones((1,), dtype=torch.float32, device=device)
    depth_beta = torch.zeros((1, 3), dtype=torch.float32, device=device)
    opacity = torch.full((1,), 0.4, dtype=torch.float32, device=device)
    feature = torch.ones((1, 1), dtype=torch.float32, device=device)
    ma_chunk = shift_ma_for_frame_chunk(
        ma,
        global_frames=global_config.frames,
        frame_start=44,
        chunk_frames=local_config.frames,
    )
    pixel_ids = torch.tensor([1 * 32 * 32 + 8 * 32 + 18], dtype=torch.int32, device=device)

    render = render_uvt_feature_sparse_pixels_with_bins(
        ma_chunk,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        pixel_ids,
        local_config,
    )
    torch.mps.synchronize()

    assert int(render.tile_counts.sum().detach().cpu()) > 0
    assert float(render.alpha_values[0].detach().cpu()) > 0.30
