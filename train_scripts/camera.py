from dataclasses import dataclass

import torch


@dataclass
class CameraSpec:
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_world: torch.Tensor


DefaultCamera = CameraSpec


def make_intrinsics(fx, fy, cx, cy, device=None, dtype=torch.float32) -> torch.Tensor:
    device = device or torch.device("cpu")
    return torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def make_identity_extrinsics(device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    device = device or torch.device("cpu")
    rotation = torch.eye(3, device=device, dtype=dtype)
    translation = torch.zeros(3, device=device, dtype=dtype)
    return rotation, translation


def make_default_camera(image_size: int, device=None, focal_scale: float = 1.0) -> CameraSpec:
    device = device or torch.device("cpu")
    camera_to_world = torch.eye(4, device=device, dtype=torch.float32)
    focal = float(image_size) * float(focal_scale)
    center = float(image_size) / 2.0
    return CameraSpec(
        fx=focal,
        fy=focal,
        cx=center,
        cy=center,
        camera_to_world=camera_to_world,
    )


def _pixel_grid(height: int, width: int, device):
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return xs, ys


def _camera_scalar_tensor(value, device, dtype=torch.float32):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(float(value), device=device, dtype=dtype)


def _normalize_vector(value, eps: float = 1e-6):
    return value / torch.linalg.norm(value).clamp_min(eps)


def build_look_at_camera_to_world(camera_position, target=None, up=None):
    device = camera_position.device
    dtype = camera_position.dtype
    if target is None:
        target = torch.zeros(3, device=device, dtype=dtype)
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    forward = _normalize_vector(target - camera_position)
    right = _normalize_vector(torch.cross(up, forward, dim=0))
    camera_up = _normalize_vector(torch.cross(forward, right, dim=0))

    camera_to_world = torch.eye(4, device=device, dtype=dtype)
    camera_to_world[:3, :3] = torch.stack([right, camera_up, forward], dim=1)
    camera_to_world[:3, 3] = camera_position
    return camera_to_world


def make_orbit_camera(image_size, radius, azimuth, elevation, focal, device=None, dtype=torch.float32) -> CameraSpec:
    if torch.is_tensor(radius):
        device = radius.device
        dtype = radius.dtype
    device = device or torch.device("cpu")

    radius = _camera_scalar_tensor(radius, device=device, dtype=dtype)
    azimuth = _camera_scalar_tensor(azimuth, device=device, dtype=dtype)
    elevation = _camera_scalar_tensor(elevation, device=device, dtype=dtype)
    focal = _camera_scalar_tensor(focal, device=device, dtype=dtype)
    image_extent = _camera_scalar_tensor(float(image_size), device=device, dtype=dtype)

    cos_elevation = torch.cos(elevation)
    position = torch.stack(
        [
            radius * cos_elevation * torch.sin(azimuth),
            radius * torch.sin(elevation),
            -radius * cos_elevation * torch.cos(azimuth),
        ]
    )
    center = image_extent / 2.0
    return CameraSpec(
        fx=focal,
        fy=focal,
        cx=center,
        cy=center,
        camera_to_world=build_look_at_camera_to_world(position),
    )


def build_camera_rays(camera: CameraSpec, height: int, width: int, device=None):
    device = device or camera.camera_to_world.device
    xs, ys = _pixel_grid(height, width, device)
    fx = _camera_scalar_tensor(camera.fx, device=device)
    fy = _camera_scalar_tensor(camera.fy, device=device)
    cx = _camera_scalar_tensor(camera.cx, device=device)
    cy = _camera_scalar_tensor(camera.cy, device=device)
    x = (xs - cx) / fx
    y = (ys - cy) / fy

    dirs_camera = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    dirs_camera = torch.nn.functional.normalize(dirs_camera, dim=-1)

    rotation = camera.camera_to_world[:3, :3].to(device=device, dtype=torch.float32)
    origin = camera.camera_to_world[:3, 3].to(device=device, dtype=torch.float32)

    dirs_world = torch.einsum("ij,hwj->hwi", rotation, dirs_camera)
    dirs_world = torch.nn.functional.normalize(dirs_world, dim=-1)
    origins_world = origin.view(1, 1, 3).expand(height, width, 3)
    return origins_world, dirs_world


def build_camera_rays_batch(cameras: list[CameraSpec], height: int, width: int, device=None):
    if not cameras:
        raise ValueError("Expected at least one camera")

    device = device or cameras[0].camera_to_world.device
    xs, ys = _pixel_grid(height, width, device)

    fx = torch.stack([_camera_scalar_tensor(camera.fx, device=device) for camera in cameras], dim=0).view(-1, 1, 1)
    fy = torch.stack([_camera_scalar_tensor(camera.fy, device=device) for camera in cameras], dim=0).view(-1, 1, 1)
    cx = torch.stack([_camera_scalar_tensor(camera.cx, device=device) for camera in cameras], dim=0).view(-1, 1, 1)
    cy = torch.stack([_camera_scalar_tensor(camera.cy, device=device) for camera in cameras], dim=0).view(-1, 1, 1)

    x = (xs.unsqueeze(0) - cx) / fx
    y = (ys.unsqueeze(0) - cy) / fy
    dirs_camera = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    dirs_camera = torch.nn.functional.normalize(dirs_camera, dim=-1)

    rotations = torch.stack(
        [camera.camera_to_world[:3, :3].to(device=device, dtype=torch.float32) for camera in cameras],
        dim=0,
    )
    origins = torch.stack(
        [camera.camera_to_world[:3, 3].to(device=device, dtype=torch.float32) for camera in cameras],
        dim=0,
    )

    dirs_world = torch.einsum("bij,bhwj->bhwi", rotations, dirs_camera)
    dirs_world = torch.nn.functional.normalize(dirs_world, dim=-1)
    origins_world = origins[:, None, None, :].expand(-1, height, width, -1)
    return origins_world, dirs_world


def build_camera_ray_grid(camera: CameraSpec, image_size: int, device=None, normalize: bool = True):
    origins_world, dirs_world = build_camera_rays(camera, image_size, image_size, device=device)
    if normalize:
        dirs_world = torch.nn.functional.normalize(dirs_world, dim=-1)
    return origins_world, dirs_world


def build_plucker_ray_grid(camera: CameraSpec, image_size: int, device=None, channels_first: bool = False):
    origins_world, dirs_world = build_camera_rays(camera, image_size, image_size, device=device)
    moments = torch.cross(origins_world, dirs_world, dim=-1)
    plucker = torch.cat([dirs_world, moments], dim=-1)
    if channels_first:
        return plucker.permute(2, 0, 1).unsqueeze(0).contiguous()
    return plucker.contiguous()


def build_plucker_ray_grid_batch(cameras: list[CameraSpec], image_size: int, device=None, channels_first: bool = False):
    origins_world, dirs_world = build_camera_rays_batch(cameras, image_size, image_size, device=device)
    moments = torch.cross(origins_world, dirs_world, dim=-1)
    plucker = torch.cat([dirs_world, moments], dim=-1)
    if channels_first:
        return plucker.permute(0, 3, 1, 2).contiguous()
    return plucker.contiguous()
