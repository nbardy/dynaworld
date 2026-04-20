import torch


def build_pixel_grid(H, W, device):
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([gx, gy], dim=-1)


def _camera_scalar_tensor(value, device, dtype):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(float(value), device=device, dtype=dtype)


def transform_world_to_camera(means3D, cov3D, camera_to_world):
    if camera_to_world is None:
        return means3D, cov3D

    rotation_cw = camera_to_world[:3, :3].to(device=means3D.device, dtype=means3D.dtype)
    translation = camera_to_world[:3, 3].to(device=means3D.device, dtype=means3D.dtype)
    means_camera = (means3D - translation.unsqueeze(0)) @ rotation_cw
    cov_camera = rotation_cw.transpose(0, 1).unsqueeze(0) @ cov3D @ rotation_cw.unsqueeze(0)
    return means_camera, cov_camera


def project_gaussians_2d(means3D, scales, quats, opacities, rgbs, fx, fy, cx, cy, camera_to_world=None):
    fx = _camera_scalar_tensor(fx, device=means3D.device, dtype=means3D.dtype)
    fy = _camera_scalar_tensor(fy, device=means3D.device, dtype=means3D.dtype)
    cx = _camera_scalar_tensor(cx, device=means3D.device, dtype=means3D.dtype)
    cy = _camera_scalar_tensor(cy, device=means3D.device, dtype=means3D.dtype)

    qr, qi, qj, qk = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    rotation = means3D.new_zeros((means3D.shape[0], 3, 3))
    rotation[:, 0, 0] = 1.0 - 2 * (qj**2 + qk**2)
    rotation[:, 0, 1] = 2 * (qi * qj - qr * qk)
    rotation[:, 0, 2] = 2 * (qi * qk + qr * qj)
    rotation[:, 1, 0] = 2 * (qi * qj + qr * qk)
    rotation[:, 1, 1] = 1.0 - 2 * (qi**2 + qk**2)
    rotation[:, 1, 2] = 2 * (qj * qk - qr * qi)
    rotation[:, 2, 0] = 2 * (qi * qk - qr * qj)
    rotation[:, 2, 1] = 2 * (qj * qk + qr * qi)
    rotation[:, 2, 2] = 1.0 - 2 * (qi**2 + qj**2)

    scale_matrix = means3D.new_zeros((means3D.shape[0], 3, 3))
    scale_matrix[:, 0, 0] = scales[:, 0]
    scale_matrix[:, 1, 1] = scales[:, 1]
    scale_matrix[:, 2, 2] = scales[:, 2]

    gaussian_basis = rotation @ scale_matrix
    cov3D = gaussian_basis @ gaussian_basis.transpose(1, 2)
    means3D, cov3D = transform_world_to_camera(means3D, cov3D, camera_to_world)

    z = means3D[:, 2]
    sorted_idx = torch.argsort(z, descending=False, stable=True)
    means3D = means3D[sorted_idx]
    cov3D = cov3D[sorted_idx]
    opacities = opacities[sorted_idx]
    rgbs = rgbs[sorted_idx]

    x, y, z = means3D[:, 0], means3D[:, 1], means3D[:, 2]
    z_safe = torch.clamp(z, min=1e-4)
    front_mask = (z > 1e-4).to(opacities.dtype).unsqueeze(-1)
    opacities = opacities * front_mask

    jacobian = means3D.new_zeros((means3D.shape[0], 2, 3))
    jacobian[:, 0, 0] = fx / z_safe
    jacobian[:, 0, 2] = -(fx * x) / (z_safe**2)
    jacobian[:, 1, 1] = fy / z_safe
    jacobian[:, 1, 2] = -(fy * y) / (z_safe**2)

    cov2D = jacobian @ cov3D @ jacobian.transpose(1, 2)
    cov2D[:, 0, 0] += 0.3
    cov2D[:, 1, 1] += 0.3

    det = torch.clamp(cov2D[:, 0, 0] * cov2D[:, 1, 1] - cov2D[:, 0, 1] * cov2D[:, 1, 0], min=1e-6)
    inv_cov2D = torch.zeros_like(cov2D)
    inv_cov2D[:, 0, 0] = cov2D[:, 1, 1] / det
    inv_cov2D[:, 1, 1] = cov2D[:, 0, 0] / det
    inv_cov2D[:, 0, 1] = -cov2D[:, 0, 1] / det
    inv_cov2D[:, 1, 0] = -cov2D[:, 1, 0] / det

    means2D = means3D.new_zeros((means3D.shape[0], 2))
    means2D[:, 0] = (fx * x) / z_safe + cx
    means2D[:, 1] = (fy * y) / z_safe + cy

    return means2D, inv_cov2D, cov2D, opacities, rgbs
