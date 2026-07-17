from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from powerfoam_direct import (
    PowerFoamRenderOptions,
    build_power_adjacency,
    camera_facing_quaternion,
    quaternion_frames,
    render_powerfoam_torch,
)

try:
    from .report_artifacts import write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import write_report_json


UPSTREAM_POWERFOAM_COMMIT = "96392252ebd0059fe6ca98881b62e12295d9242f"


def tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "values": tensor.detach().cpu().tolist(),
    }


def make_fixture() -> dict[str, Any]:
    points = torch.tensor(
        [
            [
                [-0.18, -0.08, 1.65],
                [0.22, 0.02, 1.92],
                [0.03, 0.18, 2.22],
            ]
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor([[0.42, 0.38, 0.44]], dtype=torch.float32)
    densities = torch.tensor([[2.4, 2.0, 1.7]], dtype=torch.float32)
    quaternions = camera_facing_quaternion(frame_count=1, cell_count=3)
    normals, tangents, bitangents = quaternion_frames(quaternions)
    local_texel_sites = torch.tensor(
        [
            [
                [[-0.22, -0.15], [0.20, 0.16]],
                [[-0.18, 0.18], [0.22, -0.12]],
                [[-0.20, 0.10], [0.18, -0.18]],
            ]
        ],
        dtype=torch.float32,
    )
    texel_sites = (
        points[:, :, None, :]
        + radii[:, :, None, None]
        * (
            local_texel_sites[..., 0:1] * tangents[:, :, None, :]
            + local_texel_sites[..., 1:2] * bitangents[:, :, None, :]
        )
    )
    texel_height = radii[:, :, None] * torch.tensor(
        [[[0.04, -0.02], [0.02, 0.03], [-0.03, 0.01]]],
        dtype=torch.float32,
    )
    axis_seed = torch.tensor(
        [
            [
                [[[0.0, 0.0, 1.0], [0.5, -0.1, 1.0]], [[-0.2, 0.1, 1.0], [0.4, 0.2, 1.0]]],
                [[[0.1, 0.2, 1.0], [-0.4, 0.1, 1.0]], [[0.3, -0.2, 1.0], [-0.1, -0.3, 1.0]]],
                [[[-0.3, 0.0, 1.0], [0.2, 0.4, 1.0]], [[0.1, -0.4, 1.0], [-0.5, 0.2, 1.0]]],
            ]
        ],
        dtype=torch.float32,
    )
    texel_sv_axis = 12.0 * F.normalize(axis_seed, dim=-1)
    texel_sv_rgb = torch.tensor(
        [
            [
                [[[0.34, -0.16, -0.22], [-0.08, 0.20, -0.14]], [[0.10, -0.18, 0.22], [-0.20, 0.16, 0.08]]],
                [[[-0.24, 0.30, -0.10], [0.20, -0.08, 0.12]], [[-0.12, 0.08, 0.26], [0.18, 0.18, -0.18]]],
                [[[0.08, 0.04, 0.30], [-0.28, 0.20, 0.08]], [[0.24, -0.20, 0.14], [-0.16, -0.10, 0.28]]],
            ]
        ],
        dtype=torch.float32,
    )
    adjacency = build_power_adjacency(points, radii, neighbor_count=2, mode="cech_aabb")
    origin = torch.tensor([0.16, -0.08, 0.0], dtype=torch.float32)
    targets = torch.tensor(
        [
            [[-0.12, -0.10, 1.8], [0.16, -0.06, 1.95]],
            [[-0.10, 0.12, 2.05], [0.14, 0.14, 2.20]],
        ],
        dtype=torch.float32,
    )
    directions = F.normalize(targets - origin.view(1, 1, 3), dim=-1)
    origins = origin.view(1, 1, 3).expand_as(directions)
    rays = torch.cat([origins, directions], dim=-1).unsqueeze(0)
    options = PowerFoamRenderOptions(max_alpha=0.97, texel_temperature=9.0)
    expected = render_powerfoam_torch(
        points,
        radii,
        densities,
        normals,
        texel_sites,
        None,
        texel_height,
        adjacency,
        rays,
        options,
        texel_sv_axis=texel_sv_axis,
        texel_sv_rgb=texel_sv_rgb,
    )
    return {
        "metadata": {
            "name": "powerfoam_tiny_height_sv_origin_parity_v1",
            "upstream_powerfoam_commit": UPSTREAM_POWERFOAM_COMMIT,
            "local_reference": "src/train/powerfoam_direct.py:render_powerfoam_torch",
            "notes": "Deterministic local parity fixture with nonzero ray origin; expected tensors are local Torch reference outputs.",
        },
        "render_options": {
            "near_plane": options.near_plane,
            "alpha_threshold": options.alpha_threshold,
            "transmittance_threshold": options.transmittance_threshold,
            "max_alpha": options.max_alpha,
            "eps": options.eps,
            "texel_temperature": options.texel_temperature,
            "background": list(options.background),
        },
        "inputs": {
            "points": tensor_payload(points),
            "radii": tensor_payload(radii),
            "densities": tensor_payload(densities),
            "quaternions": tensor_payload(quaternions),
            "normals": tensor_payload(normals),
            "local_texel_sites": tensor_payload(local_texel_sites),
            "texel_sites": tensor_payload(texel_sites),
            "texel_height": tensor_payload(texel_height),
            "texel_sv_axis": tensor_payload(texel_sv_axis),
            "texel_sv_rgb": tensor_payload(texel_sv_rgb),
            "adjacency": tensor_payload(adjacency),
            "rays": tensor_payload(rays),
        },
        "expected": {
            "rendered": tensor_payload(expected.rendered),
            "alpha": tensor_payload(expected.alpha),
            "normal_distance": tensor_payload(expected.normal_distance),
            "contrib": tensor_payload(expected.contrib),
            "visible_mask": tensor_payload(expected.visible_mask.to(dtype=torch.int32)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_origin_parity_v1.json"),
    )
    args = parser.parse_args()
    write_report_json(args.output, make_fixture())
    print(args.output)


if __name__ == "__main__":
    main()
