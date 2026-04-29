"""SSIM in pure numpy + OpenCV (no scikit-image dep).

Standard Wang/Bovik formulation: 11x11 Gaussian window, σ=1.5, K1=0.01, K2=0.03.
Returns the mean SSIM over the image.

Run as a script for ad-hoc comparisons:
    python ssim.py reference.png candidate.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def _gaussian(img: np.ndarray) -> np.ndarray:
    """11x11 Gaussian, σ=1.5 — the canonical SSIM smoothing kernel."""
    return cv2.GaussianBlur(img, (11, 11), sigmaX=1.5, sigmaY=1.5)


def ssim_channel(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """SSIM on a single 2-D float32 channel in [0, 1]. Returns (mean, map)."""
    K1, K2, L = 0.01, 0.03, 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu_x = _gaussian(x)
    mu_y = _gaussian(y)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _gaussian(x * x) - mu_x2
    sigma_y2 = _gaussian(y * y) - mu_y2
    sigma_xy = _gaussian(x * y) - mu_xy

    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    smap = num / den
    return float(smap.mean()), smap


def ssim_rgb(ref_path: Path, cand_path: Path) -> dict:
    """Compute mean RGB SSIM, resizing candidate to reference if needed.

    Resize uses INTER_AREA (high-quality downsample) when shrinking, INTER_CUBIC
    when upsampling. Both images converted to float32 in [0, 1].
    """
    ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    cand_bgr = cv2.imread(str(cand_path), cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise FileNotFoundError(ref_path)
    if cand_bgr is None:
        raise FileNotFoundError(cand_path)

    rh, rw = ref_bgr.shape[:2]
    ch, cw = cand_bgr.shape[:2]
    if (cw, ch) != (rw, rh):
        interp = cv2.INTER_AREA if (cw * ch) > (rw * rh) else cv2.INTER_CUBIC
        cand_bgr = cv2.resize(cand_bgr, (rw, rh), interpolation=interp)

    ref = ref_bgr.astype(np.float32) / 255.0
    cand = cand_bgr.astype(np.float32) / 255.0

    per_channel = []
    for c in range(3):
        s, _ = ssim_channel(ref[:, :, c], cand[:, :, c])
        per_channel.append(s)

    # Also a luma-only number for a cleaner single value.
    ref_y = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    cand_y = cv2.cvtColor(cand_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    luma_ssim, _ = ssim_channel(ref_y, cand_y)

    return {
        "ssim_luma": luma_ssim,
        "ssim_rgb_mean": float(np.mean(per_channel)),
        "ssim_b": per_channel[0],
        "ssim_g": per_channel[1],
        "ssim_r": per_channel[2],
        "reference_size": (rw, rh),
        "candidate_size_original": (cw, ch),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: ssim.py reference.png candidate.png", file=sys.stderr)
        sys.exit(1)
    out = ssim_rgb(Path(sys.argv[1]), Path(sys.argv[2]))
    for k, v in out.items():
        if isinstance(v, float):
            print("%s: %.4f" % (k, v))
        else:
            print("%s: %s" % (k, v))
