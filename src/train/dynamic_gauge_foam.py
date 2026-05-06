from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CameraBatch:
    cam_to_world: torch.Tensor
    fx: torch.Tensor
    fy: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor


@dataclass
class EvaluatedGaugeFoam:
    centers: torch.Tensor
    rotations: torch.Tensor
    radii: torch.Tensor
    opacities: torch.Tensor
    atlas: torch.Tensor


@dataclass
class GaugeFoamRenderOutput:
    rgb: torch.Tensor
    features: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    normals: torch.Tensor
    view_dirs: torch.Tensor


def logit_clamped(x: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def inverse_softplus(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    x = x.clamp_min(eps)
    return torch.log(torch.expm1(x).clamp_min(eps))


def skew(v: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack(
        [
            torch.stack([zero, -vz, vy], dim=-1),
            torch.stack([vz, zero, -vx], dim=-1),
            torch.stack([-vy, vx, zero], dim=-1),
        ],
        dim=-2,
    )


def so3_exp(omega: torch.Tensor, eps: float = 1.0e-7) -> torch.Tensor:
    theta_raw = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta = theta_raw.clamp_min(eps)
    unit_k = skew(omega / theta)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(*omega.shape[:-1], 3, 3)
    theta_m = theta[..., None]
    rot = eye + torch.sin(theta_m) * unit_k + (1.0 - torch.cos(theta_m)) * (unit_k @ unit_k)

    k_un = skew(omega)
    small_rot = eye + k_un + 0.5 * (k_un @ k_un)
    small = (theta_raw[..., 0] < 1.0e-4)[..., None, None]
    return torch.where(small, small_rot, rot)


def se3_exp(xi: torch.Tensor, eps: float = 1.0e-7) -> tuple[torch.Tensor, torch.Tensor]:
    omega = xi[..., :3]
    v = xi[..., 3:]
    theta_raw = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta = theta_raw.clamp_min(eps)
    rot = so3_exp(omega, eps=eps)

    unit_k = skew(omega / theta)
    eye = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(*xi.shape[:-1], 3, 3)
    theta_m = theta[..., None]
    v_matrix = (
        eye
        + ((1.0 - torch.cos(theta_m)) / theta_m) * unit_k
        + ((theta_m - torch.sin(theta_m)) / theta_m) * (unit_k @ unit_k)
    )

    k_un = skew(omega)
    small_v = eye + 0.5 * k_un + (1.0 / 6.0) * (k_un @ k_un)
    small = (theta_raw[..., 0] < 1.0e-4)[..., None, None]
    v_matrix = torch.where(small, small_v, v_matrix)
    return rot, (v_matrix @ v[..., None])[..., 0]


def linear_temporal_basis(t: torch.Tensor, num_ctrl: int) -> torch.Tensor:
    if num_ctrl == 1:
        return torch.ones(t.shape[0], 1, device=t.device, dtype=t.dtype)
    u = t.clamp(0.0, 1.0) * float(num_ctrl - 1)
    i0 = torch.floor(u).long().clamp(0, num_ctrl - 1)
    i1 = (i0 + 1).clamp(0, num_ctrl - 1)
    w1 = (u - i0.to(u.dtype)).clamp(0.0, 1.0)
    w0 = 1.0 - w1
    basis = torch.zeros(t.shape[0], num_ctrl, device=t.device, dtype=t.dtype)
    basis.scatter_add_(1, i0[:, None], w0[:, None])
    basis.scatter_add_(1, i1[:, None], w1[:, None])
    return basis


class ColorMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, *, rgb_skip: bool = True, residual_scale: float = 0.25):
        super().__init__()
        self.rgb_skip = bool(rgb_skip)
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        if self.rgb_skip and feature_dim >= 3:
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(
        self,
        feature_img: torch.Tensor,
        view_dir: torch.Tensor,
        normal_img: torch.Tensor,
        time_img: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.net(torch.cat([feature_img, view_dir, normal_img, time_img], dim=-1))
        if self.rgb_skip and feature_img.shape[-1] >= 3:
            return torch.sigmoid(logit_clamped(feature_img[..., :3]) + self.residual_scale * residual)
        return torch.sigmoid(residual)


def make_pinhole_camera_batch(
    *,
    batch_size: int,
    height: int,
    width: int,
    fov_degrees: float,
    device: torch.device,
    dtype: torch.dtype,
) -> CameraBatch:
    focal = 0.5 * float(height) / math.tan(math.radians(float(fov_degrees)) * 0.5)
    cam_to_world = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    return CameraBatch(
        cam_to_world=cam_to_world,
        fx=torch.full((batch_size,), focal, device=device, dtype=dtype),
        fy=torch.full((batch_size,), focal, device=device, dtype=dtype),
        cx=torch.full((batch_size,), float(width) * 0.5, device=device, dtype=dtype),
        cy=torch.full((batch_size,), float(height) * 0.5, device=device, dtype=dtype),
    )


def make_camera_rays(
    camera: CameraBatch,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = camera.cam_to_world.shape[0]
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs.reshape(-1)[None, :].expand(batch_size, -1)
    ys = ys.reshape(-1)[None, :].expand(batch_size, -1)
    x_cam = (xs + 0.5 - camera.cx[:, None]) / camera.fx[:, None]
    y_cam = (ys + 0.5 - camera.cy[:, None]) / camera.fy[:, None]
    dirs_cam = F.normalize(torch.stack([x_cam, y_cam, torch.ones_like(x_cam)], dim=-1), dim=-1)
    rot = camera.cam_to_world[:, :3, :3]
    origins = camera.cam_to_world[:, :3, 3][:, None, :].expand(-1, height * width, -1)
    dirs = F.normalize(torch.einsum("bij,bpj->bpi", rot, dirs_cam), dim=-1)
    return origins, dirs


def sample_atlas_bilinear(atlas: torch.Tensor, prim_idx: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    _, feature_dim, atlas_res, _ = atlas.shape
    pixel_count, hit_count = prim_idx.shape
    atlas_nhwf = atlas.permute(0, 2, 3, 1).contiguous()
    x = ((uv[..., 0] * 0.5 + 0.5) * float(atlas_res - 1)).clamp(0.0, float(atlas_res - 1))
    y = ((uv[..., 1] * 0.5 + 0.5) * float(atlas_res - 1)).clamp(0.0, float(atlas_res - 1))
    x0 = torch.floor(x).long().clamp(0, atlas_res - 1)
    y0 = torch.floor(y).long().clamp(0, atlas_res - 1)
    x1 = (x0 + 1).clamp(0, atlas_res - 1)
    y1 = (y0 + 1).clamp(0, atlas_res - 1)
    wx = (x - x0.to(x.dtype))[..., None]
    wy = (y - y0.to(y.dtype))[..., None]

    prim_flat = prim_idx.reshape(-1)
    f00 = atlas_nhwf[prim_flat, y0.reshape(-1), x0.reshape(-1)].reshape(pixel_count, hit_count, feature_dim)
    f10 = atlas_nhwf[prim_flat, y0.reshape(-1), x1.reshape(-1)].reshape(pixel_count, hit_count, feature_dim)
    f01 = atlas_nhwf[prim_flat, y1.reshape(-1), x0.reshape(-1)].reshape(pixel_count, hit_count, feature_dim)
    f11 = atlas_nhwf[prim_flat, y1.reshape(-1), x1.reshape(-1)].reshape(pixel_count, hit_count, feature_dim)
    return (1.0 - wx) * (1.0 - wy) * f00 + wx * (1.0 - wy) * f10 + (1.0 - wx) * wy * f01 + wx * wy * f11


def render_dynamic_gauge_foam(
    foam: EvaluatedGaugeFoam,
    camera: CameraBatch,
    times: torch.Tensor,
    color_mlp: ColorMLP,
    *,
    height: int,
    width: int,
    chunk_pixels: int,
    max_hits: int,
    near: float,
    far: float,
    falloff: float,
    min_alpha: float,
    background_feature: float = 0.0,
) -> GaugeFoamRenderOutput:
    device = foam.centers.device
    dtype = foam.centers.dtype
    batch_size, primitive_count, _ = foam.centers.shape
    feature_dim = foam.atlas.shape[1]
    total_pixels = height * width
    origins, dirs = make_camera_rays(camera, height, width, device, dtype)

    all_features = []
    all_alpha = []
    all_depth = []
    all_normals = []
    all_views = []
    radii2 = foam.radii.square().clamp_min(1.0e-8)

    for batch in range(batch_size):
        centers = foam.centers[batch]
        rotations_t = foam.rotations[batch].transpose(-1, -2)
        prim_normals = foam.rotations[batch, :, :, 2]
        for start in range(0, total_pixels, int(chunk_pixels)):
            end = min(total_pixels, start + int(chunk_pixels))
            origin = origins[batch, start:end]
            direction = dirs[batch, start:end]
            pixel_count = origin.shape[0]

            rel_origin = origin[:, None, :] - centers[None, :, :]
            origin_local = torch.einsum("nij,pnj->pni", rotations_t, rel_origin)
            direction_local = torch.einsum("nij,pj->pni", rotations_t, direction)
            denom = direction_local[..., 2]
            hit_t = -origin_local[..., 2] / (denom + 1.0e-8)
            xy = origin_local[..., :2] + hit_t[..., None] * direction_local[..., :2]
            rho2 = (xy.square().sum(dim=-1) / radii2[None, :]).clamp_min(0.0)
            valid = (denom.abs() > 1.0e-6) & (hit_t > float(near)) & (hit_t < float(far)) & (rho2 <= 1.0)
            alpha = foam.opacities[None, :] * torch.exp(-float(falloff) * rho2)
            alpha = torch.where(valid, alpha, torch.zeros_like(alpha)).clamp(0.0, 0.999)
            valid_alpha = alpha > float(min_alpha)
            masked_t = torch.where(valid_alpha, hit_t, torch.full_like(hit_t, float(far)))
            hit_count = min(int(max_hits), primitive_count)
            sorted_t, hit_idx = torch.topk(masked_t, k=hit_count, dim=-1, largest=False, sorted=True)
            hit_valid = sorted_t < float(far)
            hit_alpha = torch.gather(alpha, 1, hit_idx) * hit_valid.to(dtype)
            hit_xy = torch.gather(xy, 1, hit_idx[..., None].expand(-1, -1, 2))
            hit_uv = hit_xy / foam.radii[hit_idx].clamp_min(1.0e-6)[..., None]
            hit_features = sample_atlas_bilinear(foam.atlas, hit_idx, hit_uv)

            one_minus = (1.0 - hit_alpha).clamp(0.0, 1.0)
            trans_before = torch.cumprod(
                torch.cat([torch.ones(pixel_count, 1, device=device, dtype=dtype), one_minus[:, :-1]], dim=1),
                dim=1,
            )
            weights = trans_before * hit_alpha
            features = (weights[..., None] * hit_features).sum(dim=1)
            alpha_out = weights.sum(dim=1, keepdim=True)
            depth = (weights * sorted_t).sum(dim=1, keepdim=True) / alpha_out.clamp_min(1.0e-6)
            hit_normals = prim_normals[hit_idx]
            normals = F.normalize((weights[..., None] * hit_normals).sum(dim=1), dim=-1, eps=1.0e-6)
            normals = torch.where(alpha_out > 1.0e-6, normals, torch.zeros_like(normals))
            if float(background_feature) != 0.0:
                features = features + (1.0 - alpha_out) * float(background_feature)
            all_features.append(features)
            all_alpha.append(alpha_out)
            all_depth.append(depth)
            all_normals.append(normals)
            all_views.append(F.normalize(-direction, dim=-1))

    features_img = torch.cat(all_features, dim=0).reshape(batch_size, height, width, feature_dim)
    alpha_img = torch.cat(all_alpha, dim=0).reshape(batch_size, height, width, 1)
    depth_img = torch.cat(all_depth, dim=0).reshape(batch_size, height, width, 1)
    normal_img = torch.cat(all_normals, dim=0).reshape(batch_size, height, width, 3)
    view_img = torch.cat(all_views, dim=0).reshape(batch_size, height, width, 3)
    time_img = times[:, None, None, None].expand(batch_size, height, width, 1)
    rgb = color_mlp(features_img, view_img, normal_img, time_img)
    return GaugeFoamRenderOutput(rgb=rgb, features=features_img, alpha=alpha_img, depth=depth_img, normals=normal_img, view_dirs=view_img)


def make_video_grid_init(
    frames: torch.Tensor,
    *,
    primitive_count: int,
    fov_degrees: float,
    depth: float,
    radius_scale: float,
    feature_dim: int,
    atlas_res: int,
    feature_noise: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, height, width = frames.shape
    cols = math.ceil(math.sqrt(float(primitive_count)))
    rows = math.ceil(float(primitive_count) / float(cols))
    xs01 = (torch.arange(cols, dtype=torch.float32) + 0.5) / float(cols)
    ys01 = (torch.arange(rows, dtype=torch.float32) + 0.5) / float(rows)
    yy01, xx01 = torch.meshgrid(ys01, xs01, indexing="ij")
    uv01 = torch.stack([xx01.reshape(-1), yy01.reshape(-1)], dim=-1)[:primitive_count]

    half_y = math.tan(math.radians(float(fov_degrees)) * 0.5) * float(depth)
    half_x = half_y * (float(width) / float(height))
    x_world = (uv01[:, 0] * 2.0 - 1.0) * half_x
    y_world = (uv01[:, 1] * 2.0 - 1.0) * -half_y
    points = torch.stack([x_world, y_world, torch.full_like(x_world, float(depth))], dim=-1)
    cell_w = 2.0 * half_x / float(cols)
    cell_h = 2.0 * half_y / float(rows)
    radii = torch.full((primitive_count,), float(radius_scale) * max(cell_w, cell_h))

    x_idx = (uv01[:, 0] * float(width - 1)).round().long().clamp(0, width - 1)
    y_idx = (uv01[:, 1] * float(height - 1)).round().long().clamp(0, height - 1)
    colors = frames[0, :, y_idx, x_idx].permute(1, 0).contiguous()
    atlas = torch.zeros(primitive_count, feature_dim, atlas_res, atlas_res)
    if feature_dim >= 3:
        atlas[:, :3] = colors[:, :, None, None]
    if feature_dim > 3 and float(feature_noise) > 0.0:
        atlas[:, 3:] = float(feature_noise) * torch.randn(
            primitive_count,
            feature_dim - 3,
            atlas_res,
            atlas_res,
            generator=generator,
        )
    return points, radii, atlas


class DynamicGaugeFoamVideo(nn.Module):
    def __init__(
        self,
        *,
        frame_times: torch.Tensor,
        init_frames: torch.Tensor,
        primitive_count: int,
        feature_dim: int,
        atlas_res: int,
        num_time_ctrl: int,
        render_size: int,
        fov_degrees: float,
        init_depth: float,
        radius_scale: float,
        opacity_init: float,
        feature_noise: float,
        color_hidden_dim: int,
        rgb_skip: bool,
        seed: int,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        points, radii, atlas = make_video_grid_init(
            init_frames,
            primitive_count=primitive_count,
            fov_degrees=fov_degrees,
            depth=init_depth,
            radius_scale=radius_scale,
            feature_dim=feature_dim,
            atlas_res=atlas_res,
            feature_noise=feature_noise,
            generator=generator,
        )
        self.p0 = nn.Parameter(points)
        self.log_radius = nn.Parameter(inverse_softplus(radii))
        opacity = torch.full((primitive_count,), float(opacity_init)).clamp(1.0e-4, 1.0 - 1.0e-4)
        self.logit_opacity = nn.Parameter(logit_clamped(opacity))
        self.twist_ctrl = nn.Parameter(torch.zeros(primitive_count, num_time_ctrl, 6))
        self.atlas = nn.Parameter(atlas)
        self.color_mlp = ColorMLP(feature_dim, hidden_dim=color_hidden_dim, rgb_skip=rgb_skip)
        self.render_size = int(render_size)
        self.fov_degrees = float(fov_degrees)
        self.num_time_ctrl = int(num_time_ctrl)
        self.register_buffer("frame_times", frame_times.reshape(-1).float(), persistent=False)
        self.register_buffer("initial_centers", points.clone(), persistent=False)
        self.register_buffer("initial_radii", radii.clone(), persistent=False)
        self.register_buffer("initial_atlas", atlas.clone(), persistent=False)

    def evaluate_times(self, times: torch.Tensor) -> EvaluatedGaugeFoam:
        basis = linear_temporal_basis(times.reshape(-1), self.num_time_ctrl)
        xi = torch.einsum("bk,nkc->bnc", basis, self.twist_ctrl)
        rotations, translations = se3_exp(xi.reshape(-1, 6))
        batch_size = times.numel()
        rotations = rotations.reshape(batch_size, self.p0.shape[0], 3, 3)
        translations = translations.reshape(batch_size, self.p0.shape[0], 3)
        return EvaluatedGaugeFoam(
            centers=self.p0[None, :, :] + translations,
            rotations=rotations,
            radii=F.softplus(self.log_radius) + 1.0e-5,
            opacities=torch.sigmoid(self.logit_opacity),
            atlas=self.atlas,
        )

    def forward(
        self,
        frame_indices: torch.Tensor,
        *,
        chunk_pixels: int,
        max_hits: int,
        near: float,
        far: float,
        falloff: float,
        min_alpha: float,
        background_feature: float = 0.0,
    ) -> GaugeFoamRenderOutput:
        times = self.frame_times[frame_indices.to(device=self.frame_times.device, dtype=torch.long)]
        foam = self.evaluate_times(times)
        camera = make_pinhole_camera_batch(
            batch_size=int(frame_indices.numel()),
            height=self.render_size,
            width=self.render_size,
            fov_degrees=self.fov_degrees,
            device=self.p0.device,
            dtype=self.p0.dtype,
        )
        return render_dynamic_gauge_foam(
            foam,
            camera,
            times,
            self.color_mlp,
            height=self.render_size,
            width=self.render_size,
            chunk_pixels=chunk_pixels,
            max_hits=max_hits,
            near=near,
            far=far,
            falloff=falloff,
            min_alpha=min_alpha,
            background_feature=background_feature,
        )

    @torch.no_grad()
    def state_metrics(self) -> dict[str, float]:
        radii = F.softplus(self.log_radius) + 1.0e-5
        center_delta = torch.linalg.vector_norm(self.p0 - self.initial_centers.to(self.p0.device), dim=-1)
        return {
            "state_mean_center_delta": float(center_delta.mean().detach().cpu()),
            "state_p95_center_delta": float(center_delta.flatten().quantile(0.95).detach().cpu()),
            "state_max_center_delta": float(center_delta.max().detach().cpu()),
            "state_mean_radius": float(radii.mean().detach().cpu()),
            "state_mean_radius_delta": float((radii - self.initial_radii.to(radii.device)).abs().mean().detach().cpu()),
            "state_mean_opacity": float(torch.sigmoid(self.logit_opacity).mean().detach().cpu()),
            "state_mean_atlas_delta": float((self.atlas - self.initial_atlas.to(self.atlas.device)).abs().mean().detach().cpu()),
            "state_mean_twist_abs": float(self.twist_ctrl.abs().mean().detach().cpu()),
        }


def build_knn_edges(points: torch.Tensor, k: int) -> torch.Tensor:
    with torch.no_grad():
        dist = torch.cdist(points.detach().cpu(), points.detach().cpu())
        idx = torch.topk(dist, k=min(k + 1, points.shape[0]), dim=-1, largest=False).indices[:, 1:]
        src = torch.arange(points.shape[0])[:, None].expand_as(idx)
        return torch.stack([src.reshape(-1), idx.reshape(-1)], dim=0)


def gauge_connection_loss(centers: torch.Tensor, rotations: torch.Tensor, centers0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    d_now = torch.linalg.vector_norm(centers[:, dst] - centers[:, src], dim=-1)
    d0 = torch.linalg.vector_norm(centers0[dst] - centers0[src], dim=-1)[None, :].detach()
    dist_loss = F.smooth_l1_loss(d_now, d0.expand_as(d_now))
    rel = rotations[:, src].transpose(-1, -2) @ rotations[:, dst]
    eye = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
    return dist_loss + 0.1 * ((rel - eye) ** 2).mean()


def temporal_accel_loss(prev_centers: torch.Tensor, centers: torch.Tensor, next_centers: torch.Tensor) -> torch.Tensor:
    return (next_centers - 2.0 * centers + prev_centers).square().mean()


def atlas_total_variation(atlas: torch.Tensor) -> torch.Tensor:
    if atlas.shape[-1] < 2 or atlas.shape[-2] < 2:
        return atlas.new_tensor(0.0)
    return (atlas[..., 1:, :] - atlas[..., :-1, :]).abs().mean() + (atlas[..., :, 1:] - atlas[..., :, :-1]).abs().mean()
