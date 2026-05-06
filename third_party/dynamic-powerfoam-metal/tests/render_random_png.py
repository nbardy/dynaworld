from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_dynamic_powerfoam_metal import FoamRasterConfig, rasterize_power_foam
from torch_dynamic_powerfoam_metal.random_scene import (
    make_adjacency,
    make_pinhole_rays,
    make_power_sorted_ids,
    make_random_foam,
)


def save_rgb(path: Path, rgb: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = rgb.detach().cpu().clamp(0.0, 1.0).numpy()
    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr8).save(path)


def save_alpha(path: Path, alpha: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = alpha.detach().cpu().clamp(0.0, 1.0).numpy()
    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr8).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--adjacency", choices=["overlap", "knn"], default="overlap")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "random_foam.png",
    )
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the DynamicPowerFoam Metal PNG smoke")
    device = torch.device("mps")
    points, radii, densities, features = make_random_foam(
        cell_count=args.cells,
        feature_dim=3,
        device=device,
        seed=args.seed,
    )
    adjacency, offsets = make_adjacency(points, radii, mode=args.adjacency, neighbors=args.neighbors)
    rays = make_pinhole_rays(batch_size=1, height=args.height, width=args.width, device=device)
    sorted_ids = make_power_sorted_ids(points, radii, rays)

    with torch.no_grad():
        features_img, alpha = rasterize_power_foam(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            rays,
            FoamRasterConfig(alpha_threshold=0.0),
            sorted_ids=sorted_ids,
        )
    rgb = features_img[0] + (1.0 - alpha[0, ..., None]) * torch.tensor([0.03, 0.035, 0.045], device=device)
    save_rgb(args.out, rgb)
    alpha_out = args.out.with_name(args.out.stem + "_alpha.png")
    save_alpha(alpha_out, alpha[0])
    avg_degree = float(adjacency.numel()) / float(args.cells) if args.cells else 0.0
    print(f"wrote {args.out}")
    print(f"wrote {alpha_out}")
    print(f"cells={args.cells} resolution={args.width}x{args.height} adjacency={args.adjacency} avg_degree={avg_degree:.2f}")


if __name__ == "__main__":
    main()
