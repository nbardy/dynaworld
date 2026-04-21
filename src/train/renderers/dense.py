import torch

from .common import MIN_RENDER_DEPTH, build_pixel_grid, project_gaussians_2d, project_gaussians_2d_batch


def _render_aux(alpha, weights):
    spatial_dims = tuple(range(alpha.ndim - 2, alpha.ndim))
    return {
        "alpha_max": alpha.detach().amax(dim=spatial_dims),
        "weight_sum": weights.detach().sum(dim=spatial_dims),
    }


def render_pytorch_3dgs(
    means3D,
    scales,
    quats,
    opacities,
    rgbs,
    H,
    W,
    fx,
    fy,
    cx,
    cy,
    grid=None,
    camera_to_world=None,
    near_plane=None,
    return_aux=False,
):
    means2D, invCov2D, _cov2D, opacities, rgbs = project_gaussians_2d(
        means3D,
        scales,
        quats,
        opacities,
        rgbs,
        fx,
        fy,
        cx,
        cy,
        camera_to_world=camera_to_world,
        near_plane=near_plane if near_plane is not None else MIN_RENDER_DEPTH,
    )

    if grid is None:
        grid = build_pixel_grid(H, W, means3D.device)

    dx = grid.view(1, H, W, 2) - means2D.view(-1, 1, 1, 2)
    invCov2D_expand = invCov2D.view(-1, 1, 1, 2, 2)
    dx_unsqueezed = dx.unsqueeze(-1)
    dx_T = dx.unsqueeze(-2)
    power = -0.5 * (dx_T @ invCov2D_expand @ dx_unsqueezed).squeeze(-1).squeeze(-1)
    power = torch.minimum(power, torch.zeros((), device=power.device, dtype=power.dtype))

    alpha = opacities.view(-1, 1, 1) * torch.exp(power)
    alpha = torch.clamp(alpha, 0.0, 0.99)

    T_map = torch.cat(
        [
            alpha.new_ones((1, H, W)),
            torch.cumprod(1.0 - alpha[:-1] + 1e-6, dim=0),
        ],
        dim=0,
    )
    weights = T_map * alpha
    render = (weights.unsqueeze(-1) * rgbs.view(-1, 1, 1, 3)).sum(dim=0)
    bg = torch.ones_like(render)
    out = render + (1.0 - weights.sum(dim=0).unsqueeze(-1).clamp(0, 1)) * bg
    out = out.clamp(0.0, 1.0).permute(2, 0, 1)
    if return_aux:
        return out, _render_aux(alpha, weights)
    return out


def render_pytorch_3dgs_batch(
    means3D,
    scales,
    quats,
    opacities,
    rgbs,
    H,
    W,
    fx,
    fy,
    cx,
    cy,
    grid=None,
    camera_to_world=None,
    near_plane=None,
    return_aux=False,
):
    means2D, invCov2D, _cov2D, opacities, rgbs = project_gaussians_2d_batch(
        means3D,
        scales,
        quats,
        opacities,
        rgbs,
        fx,
        fy,
        cx,
        cy,
        camera_to_world=camera_to_world,
        near_plane=near_plane if near_plane is not None else MIN_RENDER_DEPTH,
    )

    if grid is None:
        grid = build_pixel_grid(H, W, means3D.device)

    batch_size, gaussian_count = means2D.shape[:2]
    dx = grid.view(1, 1, H, W, 2) - means2D.view(batch_size, gaussian_count, 1, 1, 2)
    dx0 = dx[..., 0]
    dx1 = dx[..., 1]
    power = -0.5 * (
        invCov2D[..., 0, 0].view(batch_size, gaussian_count, 1, 1) * dx0.square()
        + (invCov2D[..., 0, 1] + invCov2D[..., 1, 0]).view(batch_size, gaussian_count, 1, 1) * dx0 * dx1
        + invCov2D[..., 1, 1].view(batch_size, gaussian_count, 1, 1) * dx1.square()
    )
    power = torch.minimum(power, torch.zeros((), device=power.device, dtype=power.dtype))

    alpha = opacities.squeeze(-1).view(opacities.shape[0], -1, 1, 1) * torch.exp(power)
    alpha = torch.clamp(alpha, 0.0, 0.99)

    T_map = torch.cat(
        [
            alpha.new_ones((alpha.shape[0], 1, H, W)),
            torch.cumprod(1.0 - alpha[:, :-1] + 1e-6, dim=1),
        ],
        dim=1,
    )
    weights = T_map * alpha
    render = (weights.unsqueeze(-1) * rgbs[:, :, None, None, :]).sum(dim=1)
    bg = torch.ones_like(render)
    out = render + (1.0 - weights.sum(dim=1).unsqueeze(-1).clamp(0, 1)) * bg
    out = out.clamp(0.0, 1.0).permute(0, 3, 1, 2)
    if return_aux:
        return out, _render_aux(alpha, weights)
    return out
