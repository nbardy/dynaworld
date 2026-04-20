import torch

from .common import build_pixel_grid, project_gaussians_2d


def compute_gaussian_bounds(means2D, Cov2D, opacities, H, W, bound_scale=3.0, alpha_threshold=1.0 / 255.0):
    sigma_x = torch.sqrt(torch.clamp(Cov2D[:, 0, 0], min=1e-6))
    sigma_y = torch.sqrt(torch.clamp(Cov2D[:, 1, 1], min=1e-6))
    if alpha_threshold > 0.0:
        opacity_factor = torch.sqrt(
            torch.clamp(
                2.0 * torch.log(torch.clamp(opacities.squeeze(-1), min=alpha_threshold) / alpha_threshold),
                min=0.0,
            )
        )
        radius_factor = torch.minimum(torch.full_like(opacity_factor, bound_scale), opacity_factor)
    else:
        radius_factor = torch.full_like(sigma_x, bound_scale)
    radius_x = radius_factor * sigma_x
    radius_y = radius_factor * sigma_y

    min_x = torch.floor(means2D[:, 0] - radius_x).clamp(0, W - 1).to(torch.int64)
    max_x = torch.ceil(means2D[:, 0] + radius_x).clamp(0, W - 1).to(torch.int64)
    min_y = torch.floor(means2D[:, 1] - radius_y).clamp(0, H - 1).to(torch.int64)
    max_y = torch.ceil(means2D[:, 1] + radius_y).clamp(0, H - 1).to(torch.int64)
    valid = (max_x >= min_x) & (max_y >= min_y) & (radius_factor > 0)
    return min_x, max_x, min_y, max_y, valid


def build_tile_assignments(min_x, max_x, min_y, max_y, valid, tile_size, num_tiles_x):
    tile_min_x = torch.div(min_x, tile_size, rounding_mode="floor")
    tile_max_x = torch.div(max_x, tile_size, rounding_mode="floor")
    tile_min_y = torch.div(min_y, tile_size, rounding_mode="floor")
    tile_max_y = torch.div(max_y, tile_size, rounding_mode="floor")

    widths = tile_max_x - tile_min_x + 1
    heights = tile_max_y - tile_min_y + 1
    counts = torch.where(valid, widths * heights, torch.zeros_like(widths))
    if not torch.any(counts > 0):
        return None

    max_tiles_per_gaussian = int(counts.max().item())
    local_ids = torch.arange(max_tiles_per_gaussian, device=min_x.device).view(1, -1)
    safe_widths = widths.clamp(min=1).view(-1, 1)
    local_x = local_ids % safe_widths
    local_y = torch.div(local_ids, safe_widths, rounding_mode="floor")
    assignment_valid = valid.view(-1, 1) & (local_ids < counts.view(-1, 1))

    tile_x = tile_min_x.view(-1, 1) + local_x
    tile_y = tile_min_y.view(-1, 1) + local_y
    tile_ids = tile_y * num_tiles_x + tile_x
    gaussian_ids = torch.arange(min_x.shape[0], device=min_x.device).view(-1, 1).expand_as(tile_ids)

    flat_valid = assignment_valid.reshape(-1)
    flat_tile_ids = tile_ids.reshape(-1)[flat_valid]
    flat_gaussian_ids = gaussian_ids.reshape(-1)[flat_valid]

    order = torch.argsort(flat_tile_ids, stable=True)
    tile_ids_sorted = flat_tile_ids[order]
    gaussian_ids_sorted = flat_gaussian_ids[order]
    unique_tile_ids, counts_per_tile = torch.unique_consecutive(tile_ids_sorted, return_counts=True)

    max_gaussians_per_tile = int(counts_per_tile.max().item())
    starts = torch.cumsum(counts_per_tile, dim=0) - counts_per_tile
    positions = torch.arange(max_gaussians_per_tile, device=min_x.device).view(1, -1)
    valid_positions = positions < counts_per_tile.view(-1, 1)
    safe_positions = torch.where(valid_positions, starts.view(-1, 1) + positions, torch.zeros_like(positions))
    packed_gaussian_ids = gaussian_ids_sorted[safe_positions]

    return unique_tile_ids, packed_gaussian_ids, valid_positions


def render_pytorch_3dgs_tiled(
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
    tile_size=8,
    bound_scale=3.0,
    alpha_threshold=1.0 / 255.0,
    camera_to_world=None,
):
    raw_means3D = means3D
    raw_scales = scales
    raw_quats = quats
    raw_opacities = opacities
    raw_rgbs = rgbs
    means2D, invCov2D, Cov2D, opacities, rgbs = project_gaussians_2d(
        means3D, scales, quats, opacities, rgbs, fx, fy, cx, cy, camera_to_world=camera_to_world
    )
    min_x, max_x, min_y, max_y, valid = compute_gaussian_bounds(
        means2D,
        Cov2D,
        opacities,
        H,
        W,
        bound_scale=bound_scale,
        alpha_threshold=alpha_threshold,
    )

    num_tiles_x = (W + tile_size - 1) // tile_size
    assignments = build_tile_assignments(min_x, max_x, min_y, max_y, valid, tile_size, num_tiles_x)
    if assignments is None:
        from .dense import render_pytorch_3dgs

        return render_pytorch_3dgs(
            raw_means3D,
            raw_scales,
            raw_quats,
            raw_opacities,
            raw_rgbs,
            H,
            W,
            fx,
            fy,
            cx,
            cy,
            grid=grid,
            camera_to_world=camera_to_world,
        )

    unique_tile_ids, packed_gaussian_ids, valid_positions = assignments
    padded_H = ((H + tile_size - 1) // tile_size) * tile_size
    padded_W = ((W + tile_size - 1) // tile_size) * tile_size
    tile_x0 = (unique_tile_ids % num_tiles_x) * tile_size
    tile_y0 = torch.div(unique_tile_ids, num_tiles_x, rounding_mode="floor") * tile_size

    local_grid = build_pixel_grid(tile_size, tile_size, means3D.device)
    tile_offsets = torch.stack([tile_x0, tile_y0], dim=-1).to(local_grid.dtype)
    tile_grid = local_grid.unsqueeze(0) + tile_offsets[:, None, None, :]

    tile_means = means2D[packed_gaussian_ids]
    tile_inv_cov = invCov2D[packed_gaussian_ids]
    tile_opacities = opacities[packed_gaussian_ids].squeeze(-1) * valid_positions
    tile_rgbs = rgbs[packed_gaussian_ids] * valid_positions.unsqueeze(-1)

    dx = tile_grid[:, None, :, :, :] - tile_means[:, :, None, None, :]
    power = -0.5 * torch.einsum("tmhwi,tmij,tmhwj->tmhw", dx, tile_inv_cov, dx)
    alpha = tile_opacities[:, :, None, None] * torch.exp(power)
    alpha = torch.clamp(alpha, 0.0, 0.99) * valid_positions[:, :, None, None]

    T_map = torch.cat(
        [
            alpha.new_ones((alpha.shape[0], 1, tile_size, tile_size)),
            torch.cumprod(1.0 - alpha[:, :-1] + 1e-6, dim=1),
        ],
        dim=1,
    )
    weights = T_map * alpha
    tile_render = (weights.unsqueeze(-1) * tile_rgbs[:, :, None, None, :]).sum(dim=1)
    tile_bg = torch.ones_like(tile_render)
    tile_out = tile_render + (1.0 - weights.sum(dim=1).unsqueeze(-1).clamp(0, 1)) * tile_bg

    render_padded = means3D.new_ones((padded_H, padded_W, 3))
    tile_grid_long = tile_grid.to(torch.int64)
    flat_indices = (tile_grid_long[..., 1] * padded_W + tile_grid_long[..., 0]).reshape(-1)
    render_padded.view(-1, 3)[flat_indices] = tile_out.reshape(-1, 3)
    return render_padded[:H, :W].clamp(0.0, 1.0).permute(2, 0, 1)
