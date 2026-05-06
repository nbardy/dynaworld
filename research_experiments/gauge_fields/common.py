from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import torch


DYNAWORLD_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SRC = DYNAWORLD_ROOT / "src" / "train"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from losses import ssim_per_image  # noqa: E402

def resolve_dynaworld_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return DYNAWORLD_ROOT / value


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def robust_l1(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps).mean()


def video_metrics(rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = rendered - target
    l1 = diff.abs().mean()
    mse = (diff ** 2).mean().clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    rendered_nchw = rendered.permute(0, 3, 1, 2).contiguous()
    target_nchw = target.permute(0, 3, 1, 2).contiguous()
    window_size = min(11, int(rendered.shape[1]), int(rendered.shape[2]))
    if window_size % 2 == 0:
        window_size -= 1
    ssim = ssim_per_image(
        rendered_nchw,
        target_nchw,
        window_size=max(1, window_size),
        c1=0.0001,
        c2=0.0009,
    ).mean()
    return {
        "eval_l1": float(l1.detach().cpu()),
        "eval_mse": float(mse.detach().cpu()),
        "eval_psnr": float(psnr.detach().cpu()),
        "eval_ssim": float(ssim.detach().cpu()),
    }


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def alpha_metrics(alpha: torch.Tensor) -> dict[str, float]:
    return {
        "alpha_mean": float(alpha.mean().detach().cpu()),
        "alpha_coverage_005": float((alpha > 0.05).float().mean().detach().cpu()),
        "alpha_coverage_050": float((alpha > 0.50).float().mean().detach().cpu()),
        "alpha_coverage_090": float((alpha > 0.90).float().mean().detach().cpu()),
        "alpha_hole_fraction": float((alpha < 0.05).float().mean().detach().cpu()),
        "alpha_max": float(alpha.max().detach().cpu()),
    }


def tensor_to_uint8_image(image: torch.Tensor) -> Any:
    array = (image.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    from PIL import Image

    return Image.fromarray(array)


def save_preview_strip(
    path: Path,
    target: torch.Tensor,
    rendered: torch.Tensor,
    alpha: torch.Tensor,
    max_frames: int = 4,
) -> None:
    T, H, W, _ = target.shape
    count = min(max_frames, T)
    indices = torch.linspace(0, T - 1, count).round().long().tolist()
    rows = []
    for index in indices:
        tgt = target[index]
        ren = rendered[index]
        diff = (ren - tgt).abs()
        a = alpha[index][..., None].expand(H, W, 3)
        row = torch.cat([tgt, ren, diff, a], dim=1)
        rows.append(row)

    canvas = torch.cat(rows, dim=0)
    image = tensor_to_uint8_image(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)

    legend_path = path.with_name(path.stem + "_columns.txt")
    legend_path.write_text("columns: target | render | abs_error | alpha\n")


def save_side_by_side_mp4(
    path: Path,
    target: torch.Tensor,
    rendered: torch.Tensor,
    fps: float = 4.0,
) -> None:
    import cv2

    frames = torch.cat([target, rendered], dim=2)
    frames_u8 = (frames.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    T, H, W, _ = frames_u8.shape

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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def scalar_background(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 3:
        channels = [float(channel) for channel in value]
        if max(channels) - min(channels) > 1e-6:
            raise ValueError("The toy gauge-field renderer only supports grayscale background values.")
        return channels[0]
    raise TypeError(f"Unsupported background value: {value!r}")
