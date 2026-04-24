from dataclasses import dataclass
from typing import Literal

import torch


LensModel = Literal["pinhole", "radial_tangential", "opencv_fisheye"]


@dataclass
class CameraSpec:
    """Central-camera parameters.

    Shapes:
        fx/fy/cx/cy: scalar float/tensor for a single camera.
        camera_to_world: [4, 4], with camera-frame +z as the optical axis.
        lens_model:
            pinhole: no distortion.
            radial_tangential: OpenCV/Brown-Conrady coefficients
                distortion=[k1, k2, p1, p2, k3].
            opencv_fisheye: OpenCV fisheye/Kannala-Brandt coefficients
                distortion=[k1, k2, k3, k4].
    """

    fx: float | torch.Tensor
    fy: float | torch.Tensor
    cx: float | torch.Tensor
    cy: float | torch.Tensor
    camera_to_world: torch.Tensor
    lens_model: LensModel = "pinhole"
    distortion: torch.Tensor | tuple[float, ...] | list[float] | None = None


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


def _pixel_grid(height: int, width: int, device, dtype=torch.float32, pixel_center: float = 0.0):
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + pixel_center,
        torch.arange(width, device=device, dtype=dtype) + pixel_center,
        indexing="ij",
    )
    return xs, ys


def _camera_scalar_tensor(value, device, dtype=torch.float32):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(float(value), device=device, dtype=dtype)


def _distortion_tensor(camera: CameraSpec, size: int, device, dtype) -> torch.Tensor:
    if camera.distortion is None:
        return torch.zeros(size, device=device, dtype=dtype)
    distortion = torch.as_tensor(camera.distortion, device=device, dtype=dtype).flatten()
    if distortion.numel() > size:
        raise ValueError(
            f"{camera.lens_model} expects at most {size} distortion coefficients, got {distortion.numel()}."
        )
    if distortion.numel() == size:
        return distortion
    padded = torch.zeros(size, device=device, dtype=dtype)
    padded[: distortion.numel()] = distortion
    return padded


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


def _undistort_radial_tangential(x_distorted, y_distorted, coeffs, iterations: int):
    """Invert OpenCV/Brown-Conrady distortion in normalized image coordinates."""
    k1, k2, p1, p2, k3 = coeffs
    x = x_distorted
    y = y_distorted
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x_tangent = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_tangent = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (x_distorted - x_tangent) / radial.clamp_min(1.0e-8)
        y = (y_distorted - y_tangent) / radial.clamp_min(1.0e-8)
    return x, y


def _fisheye_theta_from_radius(radius_distorted, coeffs, iterations: int):
    """Invert theta_d = theta * (1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8)."""
    k1, k2, k3, k4 = coeffs
    theta = radius_distorted.clamp(min=0.0, max=torch.pi)
    for _ in range(iterations):
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        poly = 1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8
        f = theta * poly - radius_distorted
        derivative = (
            1.0
            + 3.0 * k1 * theta2
            + 5.0 * k2 * theta4
            + 7.0 * k3 * theta6
            + 9.0 * k4 * theta8
        )
        theta = (theta - f / derivative.clamp_min(1.0e-8)).clamp(min=0.0, max=torch.pi)
    return theta


def _camera_frame_directions(
    camera: CameraSpec,
    height: int,
    width: int,
    device,
    dtype=torch.float32,
    distortion_iterations: int = 8,
    pixel_center: float = 0.0,
):
    xs, ys = _pixel_grid(height, width, device, dtype=dtype, pixel_center=pixel_center)
    fx = _camera_scalar_tensor(camera.fx, device=device, dtype=dtype)
    fy = _camera_scalar_tensor(camera.fy, device=device, dtype=dtype)
    cx = _camera_scalar_tensor(camera.cx, device=device, dtype=dtype)
    cy = _camera_scalar_tensor(camera.cy, device=device, dtype=dtype)
    x_distorted = (xs - cx) / fx
    y_distorted = (ys - cy) / fy

    if camera.lens_model == "pinhole":
        x = x_distorted
        y = y_distorted
        dirs_camera = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        return torch.nn.functional.normalize(dirs_camera, dim=-1)

    if camera.lens_model == "radial_tangential":
        coeffs = _distortion_tensor(camera, 5, device, dtype)
        x, y = _undistort_radial_tangential(
            x_distorted,
            y_distorted,
            coeffs,
            iterations=distortion_iterations,
        )
        dirs_camera = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        return torch.nn.functional.normalize(dirs_camera, dim=-1)

    if camera.lens_model == "opencv_fisheye":
        coeffs = _distortion_tensor(camera, 4, device, dtype)
        radius_squared = x_distorted * x_distorted + y_distorted * y_distorted
        radius_distorted = torch.sqrt(radius_squared.clamp_min(1.0e-16))
        theta = _fisheye_theta_from_radius(radius_distorted, coeffs, iterations=distortion_iterations)
        safe_radius = radius_distorted.clamp_min(1.0e-8)
        unit_x = x_distorted / safe_radius
        unit_y = y_distorted / safe_radius
        sin_theta = torch.sin(theta)
        dirs_camera = torch.stack(
            [
                sin_theta * unit_x,
                sin_theta * unit_y,
                torch.cos(theta),
            ],
            dim=-1,
        )
        return torch.nn.functional.normalize(dirs_camera, dim=-1)

    raise ValueError(f"Unknown lens_model: {camera.lens_model}")


def build_central_camera_rays(
    camera: CameraSpec,
    height: int,
    width: int,
    device=None,
    dtype=torch.float32,
    distortion_iterations: int = 8,
    pixel_center: float = 0.0,
):
    """Build a central-camera ray bundle.

    Returns:
        origins_world: [H, W, 3]
        dirs_world: [H, W, 3], unit length
    """
    device = device or camera.camera_to_world.device
    dirs_camera = _camera_frame_directions(
        camera,
        height,
        width,
        device=device,
        dtype=dtype,
        distortion_iterations=distortion_iterations,
        pixel_center=pixel_center,
    )

    rotation = camera.camera_to_world[:3, :3].to(device=device, dtype=dtype)
    origin = camera.camera_to_world[:3, 3].to(device=device, dtype=dtype)

    dirs_world = torch.einsum("ij,hwj->hwi", rotation, dirs_camera)
    dirs_world = torch.nn.functional.normalize(dirs_world, dim=-1)
    origins_world = origin.view(1, 1, 3).expand(height, width, 3)
    return origins_world, dirs_world


def build_pinhole_camera_rays(
    camera: CameraSpec,
    height: int,
    width: int,
    device=None,
    dtype=torch.float32,
    pixel_center: float = 0.0,
):
    """Build rays with the old simple pinhole camera math."""
    pinhole_camera = make_camera_like(camera, lens_model="pinhole", distortion=())
    return build_central_camera_rays(
        pinhole_camera,
        height,
        width,
        device=device,
        dtype=dtype,
        distortion_iterations=0,
        pixel_center=pixel_center,
    )


def build_camera_rays(
    camera: CameraSpec,
    height: int,
    width: int,
    device=None,
    dtype=torch.float32,
    distortion_iterations: int = 8,
    pixel_center: float = 0.0,
):
    """Build rays for the configured camera model."""
    return build_central_camera_rays(
        camera,
        height,
        width,
        device=device,
        dtype=dtype,
        distortion_iterations=distortion_iterations,
        pixel_center=pixel_center,
    )


def build_central_camera_rays_batch(
    cameras: list[CameraSpec],
    height: int,
    width: int,
    device=None,
    dtype=torch.float32,
    distortion_iterations: int = 8,
    pixel_center: float = 0.0,
):
    """Build a batch of central-camera ray bundles.

    Returns:
        origins_world: [B, H, W, 3]
        dirs_world: [B, H, W, 3], unit length
    """
    if not cameras:
        raise ValueError("Expected at least one camera")

    device = device or cameras[0].camera_to_world.device
    ray_bundles = [
        build_central_camera_rays(
            camera,
            height,
            width,
            device=device,
            dtype=dtype,
            distortion_iterations=distortion_iterations,
            pixel_center=pixel_center,
        )
        for camera in cameras
    ]
    origins_world = torch.stack([origins for origins, _dirs in ray_bundles], dim=0)
    dirs_world = torch.stack([dirs for _origins, dirs in ray_bundles], dim=0)
    return origins_world, dirs_world


def build_camera_rays_batch(
    cameras: list[CameraSpec],
    height: int,
    width: int,
    device=None,
    dtype=torch.float32,
    distortion_iterations: int = 8,
    pixel_center: float = 0.0,
):
    """Build rays for each camera's configured lens model."""
    return build_central_camera_rays_batch(
        cameras,
        height,
        width,
        device=device,
        dtype=dtype,
        distortion_iterations=distortion_iterations,
        pixel_center=pixel_center,
    )


def build_plucker_from_rays(origins_world: torch.Tensor, dirs_world: torch.Tensor) -> torch.Tensor:
    """Return Plucker coordinates [..., 6] from ray origins and unit directions."""
    moments = torch.cross(origins_world, dirs_world, dim=-1)
    return torch.cat([dirs_world, moments], dim=-1)


def make_camera_like(
    camera: CameraSpec,
    *,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
    camera_to_world=None,
    lens_model: LensModel | None = None,
    distortion=None,
) -> CameraSpec:
    return CameraSpec(
        fx=camera.fx if fx is None else fx,
        fy=camera.fy if fy is None else fy,
        cx=camera.cx if cx is None else cx,
        cy=camera.cy if cy is None else cy,
        camera_to_world=camera.camera_to_world if camera_to_world is None else camera_to_world,
        lens_model=camera.lens_model if lens_model is None else lens_model,
        distortion=camera.distortion if distortion is None else distortion,
    )


def build_camera_ray_grid(camera: CameraSpec, image_size: int, device=None, normalize: bool = True):
    origins_world, dirs_world = build_camera_rays(camera, image_size, image_size, device=device)
    if normalize:
        dirs_world = torch.nn.functional.normalize(dirs_world, dim=-1)
    return origins_world, dirs_world


def build_plucker_ray_grid(camera: CameraSpec, image_size: int, device=None, channels_first: bool = False):
    origins_world, dirs_world = build_camera_rays(camera, image_size, image_size, device=device)
    plucker = build_plucker_from_rays(origins_world, dirs_world)
    if channels_first:
        return plucker.permute(2, 0, 1).unsqueeze(0).contiguous()
    return plucker.contiguous()


def build_plucker_ray_grid_batch(cameras: list[CameraSpec], image_size: int, device=None, channels_first: bool = False):
    origins_world, dirs_world = build_camera_rays_batch(cameras, image_size, image_size, device=device)
    plucker = build_plucker_from_rays(origins_world, dirs_world)
    if channels_first:
        return plucker.permute(0, 3, 1, 2).contiguous()
    return plucker.contiguous()
