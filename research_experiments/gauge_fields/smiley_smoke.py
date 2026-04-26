from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
DYNAWORLD_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from train import (  # noqa: E402
    MaterialSurfelField,
    RenderConfig,
    make_fixed_pinhole_camera,
    render_sequence,
    resolve_device,
    tensor_to_uint8_image,
)


def make_smiley_materials(device: torch.device, n_face: int, n_eye: int, n_mouth: int):
    theta = 2.0 * math.pi * torch.rand(n_face, device=device)
    radius = torch.sqrt(torch.rand(n_face, device=device))
    face = torch.stack(
        [
            radius * torch.cos(theta),
            radius * torch.sin(theta),
            torch.full_like(radius, 3.0),
        ],
        dim=-1,
    )
    face_color = torch.tensor([1.0, 0.82, 0.05], device=device).expand(n_face, 3)

    eyes = []
    for center_x in (-0.35, 0.35):
        eye_theta = 2.0 * math.pi * torch.rand(n_eye, device=device)
        eye_radius = torch.sqrt(torch.rand(n_eye, device=device)) * 0.11
        eyes.append(
            torch.stack(
                [
                    center_x + eye_radius * torch.cos(eye_theta),
                    0.28 + eye_radius * torch.sin(eye_theta),
                    torch.full_like(eye_radius, 2.94),
                ],
                dim=-1,
            )
        )
    eye = torch.cat(eyes, dim=0)
    eye_color = torch.zeros(eye.shape[0], 3, device=device)

    mouth_theta = torch.linspace(math.radians(205), math.radians(335), n_mouth, device=device)
    mouth = torch.stack(
        [
            0.55 * torch.cos(mouth_theta),
            -0.05 + 0.55 * torch.sin(mouth_theta),
            torch.full_like(mouth_theta, 2.93),
        ],
        dim=-1,
    )
    mouth_color = torch.zeros(n_mouth, 3, device=device)

    x0 = torch.cat([face, eye, mouth], dim=0)
    color = torch.cat([face_color, eye_color, mouth_color], dim=0)
    detail_start = n_face
    return x0, color, detail_start


@torch.no_grad()
def build_smiley_model(device: torch.device, frames: int) -> MaterialSurfelField:
    x0, color, detail_start = make_smiley_materials(
        device=device,
        n_face=1800,
        n_eye=180,
        n_mouth=320,
    )
    model = MaterialSurfelField(
        init_x0=x0,
        num_frames=frames,
        num_basis=2,
        init_radius=0.025,
        init_color=color,
        init_alpha_logit=0.0,
    ).to(device)

    model.raw_alpha[:detail_start] = 0.4
    model.raw_alpha[detail_start:] = 1.2
    model.log_radius[:detail_start] = math.log(0.026)
    model.log_radius[detail_start:] = math.log(0.019)

    # Low-rank deformation smoke: one wave mode and one horizontal bob mode.
    model.nr_basis.zero_()
    model.nr_coeff.zero_()
    model.nr_basis[:, 0, 2] = torch.sin(3.0 * model.x0[:, 0])
    model.nr_basis[:, 1, 0] = 1.0
    for frame in range(frames):
        phase = 2.0 * math.pi * frame / max(frames, 1)
        model.nr_coeff[frame, 0] = 0.08 * math.sin(phase)
        model.nr_coeff[frame, 1] = 0.08 * math.sin(phase)

    return model


def depth_visual(depth: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    valid = alpha > 0.02
    if bool(valid.any()):
        values = depth[valid]
        lo = values.min()
        hi = values.max()
        normalized = (depth - lo) / (hi - lo).clamp_min(1e-6)
    else:
        normalized = torch.zeros_like(depth)
    return normalized[..., None].expand(*depth.shape, 3)


def save_smiley_strip(path: Path, rgb: torch.Tensor, alpha: torch.Tensor, depth: torch.Tensor) -> None:
    depth_rgb = depth_visual(depth, alpha)
    alpha_rgb = alpha[..., None].expand(*alpha.shape, 3)
    strip = torch.cat([rgb, alpha_rgb, depth_rgb], dim=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_uint8_image(strip).save(path)
    path.with_name(path.stem + "_columns.txt").write_text("columns: rgb | alpha | depth\n")


def save_rgb_mp4(path: Path, video: torch.Tensor, fps: float) -> None:
    import cv2

    frames_u8 = (video.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    _, H, W, _ = frames_u8.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (W, H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for frame in frames_u8:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a no-training smiley through the gauge-field renderer.")
    parser.add_argument("--output-dir", default="outputs/gauge_fields/smiley_smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--pixel-chunk", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = DYNAWORLD_ROOT / output_dir

    model = build_smiley_model(device=device, frames=args.frames)
    K, w2c = make_fixed_pinhole_camera(
        num_frames=args.frames,
        H=args.size,
        W=args.size,
        fov_degrees=60.0,
        device=device,
    )
    cfg = RenderConfig(
        H=args.size,
        W=args.size,
        bg=1.0,
        min_radius_px=0.75,
        max_radius_px=16.0,
        pixel_chunk=args.pixel_chunk,
    )
    rendered = render_sequence(model, K=K, w2c=w2c, cfg=cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_smiley_strip(
        output_dir / "smiley_static.png",
        rendered["rgb"][0],
        rendered["alpha"][0],
        rendered["depth"][0],
    )
    save_rgb_mp4(
        output_dir / "smiley_wave.mp4",
        video=rendered["rgb"],
        fps=12.0,
    )
    print(f"Wrote smiley smoke outputs to {output_dir}")


if __name__ == "__main__":
    main()
