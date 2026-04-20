import torch

from .common import build_pixel_grid, project_gaussians_2d


def render_pytorch_3dgs(means3D, scales, quats, opacities, rgbs, H, W, fx, fy, cx, cy, grid=None, camera_to_world=None):
    means2D, invCov2D, _cov2D, opacities, rgbs = project_gaussians_2d(
        means3D, scales, quats, opacities, rgbs, fx, fy, cx, cy, camera_to_world=camera_to_world
    )

    if grid is None:
        grid = build_pixel_grid(H, W, means3D.device)

    dx = grid.view(1, H, W, 2) - means2D.view(-1, 1, 1, 2)
    invCov2D_expand = invCov2D.view(-1, 1, 1, 2, 2)
    dx_unsqueezed = dx.unsqueeze(-1)
    dx_T = dx.unsqueeze(-2)
    power = -0.5 * (dx_T @ invCov2D_expand @ dx_unsqueezed).squeeze(-1).squeeze(-1)

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
    return out.clamp(0.0, 1.0).permute(2, 0, 1)
