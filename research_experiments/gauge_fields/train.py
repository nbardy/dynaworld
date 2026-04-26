from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Optional, Dict

import torch
import torch.nn as nn
import wandb


DYNAWORLD_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SRC = DYNAWORLD_ROOT / "src" / "train"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from config_utils import apply_defaults, load_config_file, path_or_none, resolved_config, serialize_config_value
from sequence_data import load_uncalibrated_sequence, select_window_indices
from train_logging import build_validation_video_payload, make_preview_image, make_wandb_video


DEFAULT_CONFIG_PATH = "src/train_configs/local_mac_gauge_fields_material_surfel_128_16f_512el.jsonc"

DATA_DEFAULTS = {
    "sequence_dir": "test_data",
    "frames_dir": None,
    "video_path": "test_data/test_video_small_128_4fps.mp4",
    "frame_source": "explicit_video",
    "max_frames": 0,
    "frame_indices": None,
}

MODEL_DEFAULTS = {
    "num_elements": 512,
    "num_basis": 8,
    "init_basis_std": 0.001,
    "init_coeff_std": 0.0,
    "init_depth": 3.0,
    "init_radius": 0.05,
    "init_alpha_logit": -1.2,
}

CAMERA_DEFAULTS = {
    "lens_model": "pinhole",
    "base_fov_degrees": 60.0,
}

RENDER_DEFAULTS = {
    "render_size": 128,
    "background": [1.0, 1.0, 1.0],
    "near_plane": 1e-3,
    "far_plane": 1e4,
    "min_radius_px": 0.75,
    "max_radius_px": 24.0,
    "max_alpha_per_element": 0.95,
    "pixel_chunk": 2048,
}

TRAIN_DEFAULTS = {
    "steps": 250,
    "lr": 2e-3,
    "device": "auto",
    "seed": 0,
    "frames_per_step": 1,
    "train_frame_count": 16,
}

LOSS_DEFAULTS = {
    "rgb_weight": 1.0,
    "query_weight": 0.25,
    "flow_weight": 0.0,
    "depth_weight": 0.0,
    "arap_weight": 0.05,
    "smooth_weight": 0.02,
    "mass_weight": 1e-3,
    "radius_weight": 1e-3,
}

LOGGING_DEFAULTS = {
    "log_every": 25,
    "log_to_wandb": True,
    "wandb_project": "dynaworld",
    "wandb_run_name": "gauge-fields-material-surfel-128-16f-512el",
    "wandb_tags": ["gauge-fields", "material-surfel", "128px"],
    "wandb_mode": "online",
    "output_dir": "outputs/gauge_fields/material_surfel_128_16f_512el",
}


# ----------------------------
# Utilities
# ----------------------------

@dataclass
class RenderConfig:
    H: int
    W: int
    near: float = 1e-3
    far: float = 1e4
    bg: float = 1.0
    min_radius_px: float = 0.75
    max_radius_px: float = 24.0
    max_alpha_per_element: float = 0.95
    pixel_chunk: int = 4096


def make_pixel_grid(H: int, W: int, device):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    return torch.stack([xx.reshape(-1).float(), yy.reshape(-1).float()], dim=-1)


def project_points(
    x_world: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    near: float = 1e-3,
    far: float = 1e4,
):
    """
    x_world: [N,3]
    K: [3,3]
    w2c: [4,4], camera convention: camera looks along +Z.
    Returns:
        uv: [N,2] pixel coordinates
        z: [N]
        valid: [N]
    """
    N = x_world.shape[0]
    ones = torch.ones(N, 1, device=x_world.device, dtype=x_world.dtype)
    xh = torch.cat([x_world, ones], dim=-1)

    x_cam = (xh @ w2c.T)[..., :3]
    z = x_cam[:, 2].clamp_min(1e-8)

    x_norm = x_cam / z[:, None]
    uvh = x_norm @ K.T
    uv = uvh[:, :2]

    valid = (x_cam[:, 2] > near) & (x_cam[:, 2] < far)
    return uv, x_cam[:, 2], valid


def build_knn_edges(x0: torch.Tensor, k: int = 8):
    """
    Builds canonical neighborhood edges once.
    x0: [N,3]
    Returns:
        edges: [E,2]
        rest_lengths: [E]
    """
    with torch.no_grad():
        d = torch.cdist(x0, x0)
        idx = d.topk(k=k + 1, largest=False).indices[:, 1:]  # skip self
        src = torch.arange(x0.shape[0], device=x0.device)[:, None].expand_as(idx)
        edges = torch.stack([src.reshape(-1), idx.reshape(-1)], dim=-1)
        rest = (x0[edges[:, 0]] - x0[edges[:, 1]]).norm(dim=-1)
    return edges, rest


def robust_l1(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps).mean()


# ----------------------------
# Model: transported material samples
# ----------------------------

class MaterialSurfelField(nn.Module):
    """
    Universal material elements.

    This is not free 4D splat soup:
      - x_i(t) is derived from canonical material x_i^0 and shared low-rank motion.
      - color and opacity are persistent.
      - radius is persistent.
      - there is no free covariance per frame.
    """

    def __init__(
        self,
        init_x0: torch.Tensor,   # [N,3]
        num_frames: int,
        num_basis: int = 8,
        init_radius: float = 0.015,
        init_color: Optional[torch.Tensor] = None,
        init_alpha_logit: float = -2.0,
        init_basis_std: float = 0.001,
        init_coeff_std: float = 0.0,
    ):
        super().__init__()
        N = init_x0.shape[0]
        self.N = N
        self.T = num_frames
        self.L = num_basis

        self.x0 = nn.Parameter(init_x0.clone())                     # [N,3]
        if init_color is None:
            color_logits = torch.zeros(N, 3, device=init_x0.device, dtype=init_x0.dtype)
        else:
            color = init_color.to(device=init_x0.device, dtype=init_x0.dtype).clamp(1e-4, 1.0 - 1e-4)
            color_logits = torch.logit(color)
        self.color_logits = nn.Parameter(color_logits)              # [N,3]
        self.raw_alpha = nn.Parameter(torch.full((N, 1), init_alpha_logit, device=init_x0.device))  # [N,1]
        self.log_radius = nn.Parameter(
            torch.full((N, 1), math.log(init_radius), device=init_x0.device, dtype=init_x0.dtype)
        )

        # Low-rank material deformation:
        # x_i(t) = x_i^0 + sum_l coeff[t,l] * basis[i,l,:]
        self.nr_basis = nn.Parameter(
            float(init_basis_std) * torch.randn(N, num_basis, 3, device=init_x0.device, dtype=init_x0.dtype)
        )
        self.nr_coeff = nn.Parameter(
            float(init_coeff_std) * torch.randn(num_frames, num_basis, device=init_x0.device, dtype=init_x0.dtype)
        )

    def positions(self, t: int):
        if self.L == 0:
            return self.x0
        delta = torch.einsum("nlc,l->nc", self.nr_basis, self.nr_coeff[t])
        return self.x0 + delta

    def colors(self):
        return torch.sigmoid(self.color_logits)

    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    def radius(self):
        return torch.exp(self.log_radius).clamp_min(1e-6)

    def motion_smoothness_loss(self):
        if self.L == 0:
            return self.x0.new_zeros(())
        # velocity + acceleration smoothness on shared temporal coefficients
        vel = self.nr_coeff[1:] - self.nr_coeff[:-1]
        loss = (vel ** 2).mean()

        if self.T >= 3:
            acc = self.nr_coeff[2:] - 2 * self.nr_coeff[1:-1] + self.nr_coeff[:-2]
            loss = loss + 5.0 * (acc ** 2).mean()

        # control basis magnitude to avoid free per-element teleportation
        loss = loss + 1e-2 * (self.nr_basis ** 2).mean()
        return loss

    def mass_loss(self):
        return self.alpha().mean()

    def radius_loss(self):
        # prevents huge image-space blobs
        return (self.radius() ** 2).mean()


# ----------------------------
# Differentiable soft renderer
# ----------------------------

def render_material_field(
    model: MaterialSurfelField,
    t: int,
    K: torch.Tensor,
    w2c: torch.Tensor,
    cfg: RenderConfig,
    K_next: Optional[torch.Tensor] = None,
    w2c_next: Optional[torch.Tensor] = None,
    t_next: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Pure Torch soft projected-disk renderer.

    For production, replace with a faster renderer.
    For research, this is enough to test whether the representation trains.

    Returns:
        rgb:   [H,W,3]
        alpha: [H,W]
        depth: [H,W]
        flow:  [H,W,2] if next frame/camera supplied
        xmap:  [H,W,3] alpha-normalized canonical coordinate
    """
    device = model.x0.device
    H, W = cfg.H, cfg.W
    P = H * W

    x_t = model.positions(t)                         # [N,3]
    uv, z, valid = project_points(x_t, K, w2c, cfg.near, cfg.far)

    grid = make_pixel_grid(H, W, device=device)      # [P,2]

    # Convert persistent world radius into approximate pixel radius.
    # This is a rendering footprint, not a free 3D covariance.
    fx = K[0, 0].abs().clamp_min(1e-6)
    radius_px = model.radius()[:, 0] * fx / z.abs().clamp_min(cfg.near)
    radius_px = radius_px.clamp(cfg.min_radius_px, cfg.max_radius_px)

    alpha_i = model.alpha()[:, 0]                    # [N]

    # Approximate global depth sort. This is not exact per-pixel ordering,
    # but good enough for a toy loop.
    z_sort_key = torch.where(valid, z, torch.full_like(z, cfg.far + 1.0))
    order = torch.argsort(z_sort_key, dim=0)         # near to far

    uv_s = uv[order]                                 # [N,2]
    z_s = z[order].clamp_min(cfg.near)               # [N]
    radius_s = radius_px[order]                      # [N]
    alpha_s = alpha_i[order]                         # [N]
    valid_s = valid[order]                           # [N]
    color_s = model.colors()[order]                  # [N,3]
    x0_s = model.x0[order]                           # [N,3]

    flow_s = None
    if K_next is not None and w2c_next is not None and t_next is not None:
        x_next = model.positions(t_next)
        uv_next, _, valid_next = project_points(x_next, K_next, w2c_next, cfg.near, cfg.far)
        flow_i = uv_next - uv                         # [N,2]
        flow_i = flow_i * (valid & valid_next)[:, None].float()
        flow_s = flow_i[order]

    rgb_chunks = []
    alpha_chunks = []
    depth_chunks = []
    xmap_chunks = []
    flow_chunks = []

    pixel_chunk = P if cfg.pixel_chunk <= 0 else min(cfg.pixel_chunk, P)
    for start in range(0, P, pixel_chunk):
        end = min(start + pixel_chunk, P)
        pix = grid[start:end]                         # [C,2]

        diff = pix[None, :, :] - uv_s[:, None, :]     # [N,C,2]
        dist2 = (diff ** 2).sum(dim=-1)
        kernel = torch.exp(-0.5 * dist2 / (radius_s[:, None] ** 2 + 1e-8))

        a_s = alpha_s[:, None] * kernel
        a_s = a_s.clamp(0.0, cfg.max_alpha_per_element)
        a_s = a_s * valid_s[:, None].float()

        one_minus = (1.0 - a_s).clamp(1e-5, 1.0)
        trans = torch.cumprod(
            torch.cat([torch.ones(1, end - start, device=device), one_minus[:-1]], dim=0),
            dim=0,
        )
        weights = trans * a_s                         # [N,C]

        alpha_map = weights.sum(dim=0).clamp(0.0, 1.0)
        rgb = weights.T @ color_s
        rgb = rgb + (1.0 - alpha_map)[:, None] * cfg.bg

        depth_num = (weights * z_s[:, None]).sum(dim=0)
        depth = depth_num / alpha_map.clamp_min(1e-6)

        xmap_num = weights.T @ x0_s
        xmap = xmap_num / alpha_map[:, None].clamp_min(1e-6)

        rgb_chunks.append(rgb)
        alpha_chunks.append(alpha_map)
        depth_chunks.append(depth)
        xmap_chunks.append(xmap)

        if flow_s is not None:
            flow_num = weights.T @ flow_s
            flow = flow_num / alpha_map[:, None].clamp_min(1e-6)
            flow_chunks.append(flow)

    out = {
        "rgb": torch.cat(rgb_chunks, dim=0).reshape(H, W, 3),
        "alpha": torch.cat(alpha_chunks, dim=0).reshape(H, W),
        "depth": torch.cat(depth_chunks, dim=0).reshape(H, W),
        "xmap": torch.cat(xmap_chunks, dim=0).reshape(H, W, 3),
    }

    # Render induced optical flow from material transport if requested.
    if flow_s is not None:
        out["flow"] = torch.cat(flow_chunks, dim=0).reshape(H, W, 2)

    return out


# ----------------------------
# Losses
# ----------------------------

def arap_loss(
    model: MaterialSurfelField,
    t: int,
    edges: torch.Tensor,
    rest_lengths: torch.Tensor,
):
    """
    Local distance preservation in material neighborhoods.
    This is the cloth/isometry-ish contract.
    """
    x = model.positions(t)
    d = (x[edges[:, 0]] - x[edges[:, 1]]).norm(dim=-1)
    return ((d - rest_lengths) ** 2).mean()


def flow_loss(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    alpha: torch.Tensor,
    alpha_min: float = 0.05,
):
    """
    pred_flow: [H,W,2]
    gt_flow: [H,W,2]
    alpha: [H,W]
    """
    mask = (alpha.detach() > alpha_min) & torch.isfinite(gt_flow).all(dim=-1)
    if mask.sum() == 0:
        return pred_flow.new_zeros(())
    return robust_l1(pred_flow[mask] - gt_flow[mask])


def scale_shift_depth_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    alpha: torch.Tensor,
    alpha_min: float = 0.05,
):
    """
    Monocular depth is usually only affine/scale meaningful.
    Align pred to target by detached least squares, then apply L1.
    """
    mask = (
        (alpha.detach() > alpha_min)
        & torch.isfinite(pred_depth)
        & torch.isfinite(target_depth)
        & (target_depth > 0)
    )
    if mask.sum() < 32:
        return pred_depth.new_zeros(())

    x = pred_depth[mask].reshape(-1)
    y = target_depth[mask].reshape(-1)

    A = torch.stack([x.detach(), torch.ones_like(x)], dim=-1)
    sol = torch.linalg.lstsq(A, y.detach()).solution
    a, b = sol[0], sol[1]

    aligned = a * pred_depth + b
    return robust_l1(aligned[mask] - target_depth[mask])


# ----------------------------
# Training loop
# ----------------------------

@dataclass
class TrainWeights:
    rgb: float = 1.0
    query: float = 0.25
    flow: float = 0.05
    depth: float = 0.02
    arap: float = 0.05
    smooth: float = 0.02
    mass: float = 1e-3
    radius: float = 1e-3


def train_material_surfel_field(
    video: torch.Tensor,              # [T,H,W,3], float in [0,1]
    K: torch.Tensor,                  # [3,3] or [T,3,3]
    w2c: torch.Tensor,                # [T,4,4]
    init_x0: torch.Tensor,            # [N,3]
    init_color: Optional[torch.Tensor] = None,  # [N,3]
    flow: Optional[torch.Tensor] = None,   # [T-1,H,W,2], optional
    depth_prior: Optional[torch.Tensor] = None,  # [T,H,W], optional
    num_steps: int = 2000,
    batch_size: int = 1,
    train_frame_count: int = 0,
    num_basis: int = 8,
    init_radius: float = 0.04,
    init_alpha_logit: float = -1.2,
    init_basis_std: float = 0.001,
    init_coeff_std: float = 0.0,
    lr: float = 2e-3,
    weights: TrainWeights = TrainWeights(),
    query_every: int = 4,
    render_cfg: Optional[RenderConfig] = None,
    log_every: int = 50,
):
    """
    Minimal single-video 4D material-field train loop.

    Important:
      - This is a toy loop.
      - It is O(N*H*W), so keep H/W/N small at first.
      - Good first config: H=W=64, N=256-1024, T=20-100.
    """
    device = video.device
    T, H, W, _ = video.shape
    cfg = render_cfg if render_cfg is not None else RenderConfig(H=H, W=W)
    if cfg.H != H or cfg.W != W:
        raise ValueError(f"RenderConfig size {(cfg.H, cfg.W)} does not match video size {(H, W)}.")

    model = MaterialSurfelField(
        init_x0=init_x0,
        num_frames=T,
        num_basis=num_basis,
        init_color=init_color,
        init_radius=init_radius,
        init_alpha_logit=init_alpha_logit,
        init_basis_std=init_basis_std,
        init_coeff_std=init_coeff_std,
    ).to(device)

    edges, rest_lengths = build_knn_edges(init_x0.detach(), k=8)
    edges = edges.to(device)
    rest_lengths = rest_lengths.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def get_K(t: int):
        return K[t] if K.ndim == 3 else K

    logs = []

    for step in range(num_steps):
        opt.zero_grad(set_to_none=True)

        # Context frames: sample from the same contiguous-window idea used by
        # the video-token baselines, but keep frames_per_step small for this
        # pure Torch toy renderer.
        if train_frame_count > 0 and train_frame_count < T:
            window = select_window_indices(T, train_frame_count, device=device)
            local = torch.randint(0, window.numel(), (batch_size,), device=device)
            frames = window[local]
        else:
            frames = torch.randint(0, T, (batch_size,), device=device)

        total_loss = video.new_zeros(())
        rgb_meter = 0.0

        for tb in frames.tolist():
            use_flow = flow is not None and tb < T - 1

            out = render_material_field(
                model=model,
                t=tb,
                K=get_K(tb),
                w2c=w2c[tb],
                cfg=cfg,
                K_next=get_K(tb + 1) if use_flow else None,
                w2c_next=w2c[tb + 1] if use_flow else None,
                t_next=tb + 1 if use_flow else None,
            )

            rgb_l = robust_l1(out["rgb"] - video[tb])
            rgb_meter += float(rgb_l.detach())

            loss = weights.rgb * rgb_l
            loss = loss + weights.arap * arap_loss(model, tb, edges, rest_lengths)

            if use_flow:
                loss = loss + weights.flow * flow_loss(out["flow"], flow[tb], out["alpha"])

            if depth_prior is not None:
                loss = loss + weights.depth * scale_shift_depth_loss(
                    out["depth"], depth_prior[tb], out["alpha"]
                )

            total_loss = total_loss + loss

        total_loss = total_loss / float(batch_size)

        # Omitted/query frame loss.
        # Very simple version: every few steps, render a frame not in the sampled context batch.
        # No private paths exist in this toy model, so query loss directly pressures shared material geometry.
        if weights.query > 0 and (step % query_every == 0):
            tq = torch.randint(0, T, (1,), device=device).item()
            out_q = render_material_field(
                model=model,
                t=tq,
                K=get_K(tq),
                w2c=w2c[tq],
                cfg=cfg,
            )
            query_l = robust_l1(out_q["rgb"] - video[tq])
            total_loss = total_loss + weights.query * query_l

        total_loss = total_loss + weights.smooth * model.motion_smoothness_loss()
        total_loss = total_loss + weights.mass * model.mass_loss()
        total_loss = total_loss + weights.radius * model.radius_loss()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == num_steps - 1:
            log = {
                "step": step,
                "loss": float(total_loss.detach()),
                "rgb_l1": rgb_meter / float(batch_size),
                "mass": float(model.mass_loss().detach()),
                "motion_smooth": float(model.motion_smoothness_loss().detach()),
                "radius": float(model.radius().mean().detach()),
            }
            logs.append(log)
            print(log)

    return model, logs


# ----------------------------
# Baseline video harness
# ----------------------------


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


def make_fixed_pinhole_camera(
    num_frames: int,
    H: int,
    W: int,
    fov_degrees: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    fov = math.radians(fov_degrees)
    focal = 0.5 * float(W) / math.tan(0.5 * fov)
    K = torch.tensor(
        [
            [focal, 0.0, float(W) * 0.5],
            [0.0, focal, float(H) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    w2c = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_frames, 1, 1)
    return K, w2c


def load_baseline_video(
    sequence_dir: Path,
    frames_dir: Optional[Path],
    video_path: Optional[Path],
    frame_source: str,
    render_size: int,
    max_frames: int,
    device: torch.device,
) -> torch.Tensor:
    sequence = load_uncalibrated_sequence(
        sequence_dir=sequence_dir,
        frames_dir=frames_dir,
        video_path=video_path,
        target_size=render_size,
        max_frames=max_frames,
        frame_source=frame_source,
        device=device,
    )
    return sequence.frames.permute(0, 2, 3, 1).contiguous()


def select_configured_frames(video: torch.Tensor, frame_indices: Any) -> torch.Tensor:
    if frame_indices is None:
        return video
    if not isinstance(frame_indices, list) or not frame_indices:
        raise ValueError("data.frame_indices must be a non-empty list of integer frame indices when provided.")
    indices = torch.as_tensor(frame_indices, dtype=torch.long, device=video.device)
    if bool((indices < 0).any()) or bool((indices >= video.shape[0]).any()):
        raise IndexError(f"data.frame_indices {frame_indices!r} out of range for {video.shape[0]} loaded frames.")
    return video[indices].contiguous()


def initialize_material_points_from_first_frame(
    video: torch.Tensor,
    K: torch.Tensor,
    num_elements: int,
    init_depth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, H, W, _ = video.shape
    device = video.device

    grid_x = max(1, math.ceil(math.sqrt(float(num_elements) * float(W) / float(H))))
    grid_y = max(1, math.ceil(float(num_elements) / float(grid_x)))

    xs = torch.linspace(0.5, float(W) - 0.5, grid_x, device=device)
    ys = torch.linspace(0.5, float(H) - 0.5, grid_y, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[:num_elements]

    z = torch.full((pixels.shape[0],), init_depth, device=device)
    x = (pixels[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (pixels[:, 1] - K[1, 2]) * z / K[1, 1]
    x0 = torch.stack([x, y, z], dim=-1)

    px = pixels[:, 0].round().long().clamp(0, W - 1)
    py = pixels[:, 1].round().long().clamp(0, H - 1)
    color = video[0, py, px]
    return x0, color


@torch.no_grad()
def render_sequence(
    model: MaterialSurfelField,
    K: torch.Tensor,
    w2c: torch.Tensor,
    cfg: RenderConfig,
) -> dict[str, torch.Tensor]:
    rgbs = []
    alphas = []
    depths = []
    for t in range(model.T):
        out = render_material_field(model, t=t, K=K, w2c=w2c[t], cfg=cfg)
        rgbs.append(out["rgb"])
        alphas.append(out["alpha"])
        depths.append(out["depth"])
    return {
        "rgb": torch.stack(rgbs, dim=0),
        "alpha": torch.stack(alphas, dim=0),
        "depth": torch.stack(depths, dim=0),
    }


def video_metrics(rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = rendered - target
    l1 = diff.abs().mean()
    mse = (diff ** 2).mean().clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    return {
        "eval_l1": float(l1.detach().cpu()),
        "eval_mse": float(mse.detach().cpu()),
        "eval_psnr": float(psnr.detach().cpu()),
    }


def alpha_metrics(alpha: torch.Tensor) -> dict[str, float]:
    return {
        "alpha_mean": float(alpha.mean().detach().cpu()),
        "alpha_coverage_005": float((alpha > 0.05).float().mean().detach().cpu()),
        "alpha_coverage_050": float((alpha > 0.50).float().mean().detach().cpu()),
        "alpha_max": float(alpha.max().detach().cpu()),
    }


def model_metrics(model: MaterialSurfelField) -> dict[str, float]:
    return {
        "model_alpha_mean": float(model.alpha().mean().detach().cpu()),
        "model_radius_mean": float(model.radius().mean().detach().cpu()),
        "model_radius_max": float(model.radius().max().detach().cpu()),
        "model_motion_smooth": float(model.motion_smoothness_loss().detach().cpu()),
    }


def hwc_video_to_chw(video: torch.Tensor) -> torch.Tensor:
    return video.permute(0, 3, 1, 2).contiguous()


def wandb_log_training_logs(logs: list[dict[str, float]]) -> None:
    for log in logs:
        step = int(log["step"])
        wandb.log(
            {
                "Loss": log["loss"],
                "Loss/RGBL1": log["rgb_l1"],
                "Model/Mass": log["mass"],
                "Model/MotionSmooth": log["motion_smooth"],
                "Model/RadiusMean": log["radius"],
            },
            step=step,
        )


def wandb_final_payload(
    video: torch.Tensor,
    rendered: torch.Tensor,
    metrics: dict[str, float],
    fps: float,
) -> dict[str, Any]:
    target_chw = hwc_video_to_chw(video)
    rendered_chw = hwc_video_to_chw(rendered)
    payload: dict[str, Any] = {
        "Eval/L1": metrics["eval_l1"],
        "Eval/MSE": metrics["eval_mse"],
        "Eval/PSNR": metrics["eval_psnr"],
        "Eval/AlphaMean": metrics["alpha_mean"],
        "Eval/AlphaCoverage005": metrics["alpha_coverage_005"],
        "Eval/AlphaCoverage050": metrics["alpha_coverage_050"],
        "Eval/AlphaMax": metrics["alpha_max"],
        "Model/AlphaMeanFinal": metrics["model_alpha_mean"],
        "Model/RadiusMeanFinal": metrics["model_radius_mean"],
        "Model/RadiusMaxFinal": metrics["model_radius_max"],
        "Model/MotionSmoothFinal": metrics["model_motion_smooth"],
        "Render_GT_vs_Pred": make_preview_image(
            target_chw[0],
            rendered_chw[0],
            caption="Final frame 0",
        ),
        "GT_Video": make_wandb_video(target_chw, fps),
    }
    payload.update(build_validation_video_payload(rendered_chw, target_chw, fps))
    return payload


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


def init_wandb_if_enabled(logging_cfg: dict[str, Any], cfg: dict[str, Any]) -> bool:
    if not bool(logging_cfg["log_to_wandb"]):
        return False

    mode = logging_cfg.get("wandb_mode")
    kwargs = {
        "project": logging_cfg["wandb_project"],
        "name": logging_cfg["wandb_run_name"],
        "tags": logging_cfg.get("wandb_tags"),
        "config": serialize_config_value(cfg),
    }
    if mode:
        kwargs["mode"] = mode
    wandb.init(**kwargs)
    return True


def scalar_background(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 3:
        channels = [float(channel) for channel in value]
        if max(channels) - min(channels) > 1e-6:
            raise ValueError("The toy gauge-field renderer only supports grayscale background values.")
        return channels[0]
    raise TypeError(f"Unsupported background value: {value!r}")


def gauge_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(
        config,
        sections=("data", "model", "camera", "render", "train", "losses", "logging"),
    )
    apply_defaults(cfg["data"], DATA_DEFAULTS)
    apply_defaults(cfg["model"], MODEL_DEFAULTS)
    apply_defaults(cfg["camera"], CAMERA_DEFAULTS)
    apply_defaults(cfg["render"], RENDER_DEFAULTS)
    apply_defaults(cfg["train"], TRAIN_DEFAULTS)
    apply_defaults(cfg["losses"], LOSS_DEFAULTS)
    apply_defaults(cfg["logging"], LOGGING_DEFAULTS)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit the material-coordinate gauge-field toy renderer from a Dynaworld JSONC config."
    )
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default=None, help="Override train.device from the config.")
    parser.add_argument("--steps", type=int, default=None, help="Override train.steps from the config.")
    parser.add_argument("--output-dir", default=None, help="Override logging.output_dir from the config.")
    parser.add_argument("--wandb-mode", default=None, help="Override logging.wandb_mode from the config.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging for local probes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_dynaworld_path(args.config)
    cfg = gauge_config(load_config_file(config_path))
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.output_dir is not None:
        cfg["logging"]["output_dir"] = args.output_dir
    if args.wandb_mode is not None:
        cfg["logging"]["wandb_mode"] = args.wandb_mode
    if args.no_wandb:
        cfg["logging"]["log_to_wandb"] = False

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    camera_cfg = cfg["camera"]
    render_cfg_values = cfg["render"]
    train_cfg = cfg["train"]
    loss_cfg = cfg["losses"]
    logging_cfg = cfg["logging"]

    torch.manual_seed(int(train_cfg["seed"]))

    device = resolve_device(str(train_cfg["device"]))
    sequence_dir = resolve_dynaworld_path(data_cfg["sequence_dir"])
    frames_dir = path_or_none(data_cfg["frames_dir"])
    if frames_dir is not None:
        frames_dir = resolve_dynaworld_path(frames_dir)
    video_path = path_or_none(data_cfg["video_path"])
    if video_path is not None:
        video_path = resolve_dynaworld_path(video_path)
    output_dir = resolve_dynaworld_path(logging_cfg["output_dir"])

    video = load_baseline_video(
        sequence_dir=sequence_dir,
        frames_dir=frames_dir,
        video_path=video_path,
        frame_source=str(data_cfg["frame_source"]),
        render_size=int(render_cfg_values["render_size"]),
        max_frames=int(data_cfg["max_frames"]),
        device=device,
    )
    video = select_configured_frames(video, data_cfg["frame_indices"])
    T, H, W, _ = video.shape
    K, w2c = make_fixed_pinhole_camera(
        num_frames=T,
        H=H,
        W=W,
        fov_degrees=float(camera_cfg["base_fov_degrees"]),
        device=device,
    )
    init_x0, init_color = initialize_material_points_from_first_frame(
        video=video,
        K=K,
        num_elements=int(model_cfg["num_elements"]),
        init_depth=float(model_cfg["init_depth"]),
    )

    render_cfg = RenderConfig(
        H=H,
        W=W,
        near=float(render_cfg_values["near_plane"]),
        far=float(render_cfg_values["far_plane"]),
        bg=scalar_background(render_cfg_values["background"]),
        min_radius_px=float(render_cfg_values["min_radius_px"]),
        max_radius_px=float(render_cfg_values["max_radius_px"]),
        max_alpha_per_element=float(render_cfg_values["max_alpha_per_element"]),
        pixel_chunk=int(render_cfg_values["pixel_chunk"]),
    )
    weights = TrainWeights(
        rgb=float(loss_cfg["rgb_weight"]),
        query=float(loss_cfg["query_weight"]),
        flow=float(loss_cfg["flow_weight"]),
        depth=float(loss_cfg["depth_weight"]),
        arap=float(loss_cfg["arap_weight"]),
        smooth=float(loss_cfg["smooth_weight"]),
        mass=float(loss_cfg["mass_weight"]),
        radius=float(loss_cfg["radius_weight"]),
    )

    print(
        "Gauge-field overfit "
        f"config={config_path} video={video_path} frames={T}/{data_cfg['max_frames'] or 'all'} size={H}x{W} "
        f"elements={model_cfg['num_elements']} basis={model_cfg['num_basis']} "
        f"train_frame_count={train_cfg['train_frame_count']} frames_per_step={train_cfg['frames_per_step']} "
        f"steps={train_cfg['steps']} device={device}"
    )

    wandb_enabled = init_wandb_if_enabled(logging_cfg, cfg)
    try:
        model, logs = train_material_surfel_field(
            video=video,
            K=K,
            w2c=w2c,
            init_x0=init_x0,
            init_color=init_color,
            num_steps=int(train_cfg["steps"]),
            batch_size=int(train_cfg["frames_per_step"]),
            train_frame_count=int(train_cfg["train_frame_count"]),
            num_basis=int(model_cfg["num_basis"]),
            init_radius=float(model_cfg["init_radius"]),
            init_alpha_logit=float(model_cfg["init_alpha_logit"]),
            init_basis_std=float(model_cfg["init_basis_std"]),
            init_coeff_std=float(model_cfg["init_coeff_std"]),
            lr=float(train_cfg["lr"]),
            weights=weights,
            render_cfg=render_cfg,
            log_every=int(logging_cfg["log_every"]),
        )

        rendered = render_sequence(model, K=K, w2c=w2c, cfg=render_cfg)
        metrics = {
            **video_metrics(rendered["rgb"], video),
            **alpha_metrics(rendered["alpha"]),
            **model_metrics(model),
        }
        print({"final": metrics})

        if wandb_enabled:
            wandb_log_training_logs(logs)
            wandb.log(
                wandb_final_payload(
                    video=video,
                    rendered=rendered["rgb"],
                    metrics=metrics,
                    fps=4.0,
                ),
                step=int(train_cfg["steps"]),
            )
    finally:
        if wandb_enabled:
            wandb.finish()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", serialize_config_value(cfg))
    write_json(output_dir / "logs.json", logs)
    write_json(output_dir / "metrics.json", metrics)
    torch.save(
        {
            "model": model.state_dict(),
            "K": K.detach().cpu(),
            "w2c": w2c.detach().cpu(),
            "render_config": render_cfg.__dict__,
            "config": serialize_config_value(cfg),
            "metrics": metrics,
        },
        output_dir / "checkpoint.pt",
    )
    save_preview_strip(
        output_dir / "preview.png",
        target=video,
        rendered=rendered["rgb"],
        alpha=rendered["alpha"],
    )
    save_side_by_side_mp4(
        output_dir / "side_by_side.mp4",
        target=video,
        rendered=rendered["rgb"],
        fps=4.0,
    )
    print(f"Wrote gauge-field outputs to {output_dir}")


if __name__ == "__main__":
    main()
