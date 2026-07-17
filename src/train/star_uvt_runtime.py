from __future__ import annotations

import torch

from external_paths import PROJECT_ROOT as DYNAWORLD_ROOT
from external_paths import ensure_sys_path, third_party_path
from train_devices import resolve_torch_device, sync_torch_device


STAR_UVT_ROOT = third_party_path("fast-mac-gsplat") / "variants" / "star_uvt_v0"


def ensure_star_uvt_on_path(*, include_dynaworld_root: bool = True) -> None:
    if not STAR_UVT_ROOT.exists():
        raise FileNotFoundError(f"Missing STAR UVT checkout: {STAR_UVT_ROOT}")
    paths = (DYNAWORLD_ROOT, STAR_UVT_ROOT) if include_dynaworld_root else (STAR_UVT_ROOT,)
    ensure_sys_path(*paths)


def resolve_device(name: str) -> torch.device:
    return resolve_torch_device(name, auto_cuda=True, validate_requested=True)


def sync_device(device: torch.device) -> None:
    sync_torch_device(device)


def psnr_from_loss(loss: float) -> float:
    return float(-10.0 * torch.log10(torch.tensor(max(float(loss), 1.0e-12))).item())


__all__ = [
    "DYNAWORLD_ROOT",
    "STAR_UVT_ROOT",
    "ensure_star_uvt_on_path",
    "psnr_from_loss",
    "resolve_device",
    "sync_device",
]
