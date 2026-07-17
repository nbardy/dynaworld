from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from camera import CameraSpec
from renderers.projection import project_points_camera


@dataclass(frozen=True)
class PointCloudInitialization:
    points: torch.Tensor
    colors: torch.Tensor
    source_path: Path
    source_count: int
    sampled_count: int
    normalize_mode: str
    coordinate_frame: str
    visibility_filter: str
    sample_mode: str
    filtered_count: int


PLY_SCALAR_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def resolve_point_cloud_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = (
        path / "input.ply",
        path / "point_cloud.ply",
        path / "points3D.txt",
        path / "points3D.bin",
        path / "sparse" / "0" / "points3D.txt",
        path / "sparse" / "0" / "points3D.bin",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported point cloud file found under {path}.")


def normalize_point_cloud_colors(colors: torch.Tensor | None, count: int) -> torch.Tensor:
    if colors is None:
        return torch.full((count, 3), 0.5, dtype=torch.float32)
    colors = colors.to(dtype=torch.float32)
    if colors.numel() > 0 and float(colors.max()) > 1.0:
        colors = colors / 255.0
    return colors.clamp(0.0, 1.0)


def load_ply_point_cloud(path: Path) -> tuple[torch.Tensor, torch.Tensor | None]:
    with path.open("rb") as fh:
        first = fh.readline().decode("ascii", errors="strict").strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file.")
        fmt = None
        vertex_count = None
        in_vertex = False
        vertex_properties: list[tuple[str, str]] = []
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError(f"{path} ended before PLY end_header.")
            line = raw.decode("ascii", errors="strict").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path} has list properties on vertices; unsupported for PowerFoam init.")
                vertex_properties.append((parts[2], parts[1]))
        if fmt not in {"ascii", "binary_little_endian"}:
            raise ValueError(f"{path} PLY format {fmt!r} is unsupported.")
        if vertex_count is None:
            raise ValueError(f"{path} does not declare a vertex element.")
        prop_names = [name for name, _kind in vertex_properties]
        for required in ("x", "y", "z"):
            if required not in prop_names:
                raise ValueError(f"{path} vertex properties must include x/y/z.")
        xyz_rows = []
        rgb_rows = []
        has_rgb = all(name in prop_names for name in ("red", "green", "blue"))
        if fmt == "ascii":
            for _ in range(vertex_count):
                values = fh.readline().decode("ascii", errors="strict").split()
                if len(values) < len(vertex_properties):
                    raise ValueError(f"{path} has a short ASCII vertex row.")
                row = {name: float(values[index]) for index, (name, _kind) in enumerate(vertex_properties)}
                xyz_rows.append([row["x"], row["y"], row["z"]])
                if has_rgb:
                    rgb_rows.append([row["red"], row["green"], row["blue"]])
        else:
            try:
                row_struct = struct.Struct("<" + "".join(PLY_SCALAR_FORMATS[kind] for _name, kind in vertex_properties))
            except KeyError as exc:
                raise ValueError(f"{path} has unsupported PLY scalar type {exc.args[0]!r}.") from exc
            name_to_index = {name: index for index, (name, _kind) in enumerate(vertex_properties)}
            for _ in range(vertex_count):
                payload = fh.read(row_struct.size)
                if len(payload) != row_struct.size:
                    raise ValueError(f"{path} ended inside a binary vertex row.")
                values = row_struct.unpack(payload)
                xyz_rows.append([values[name_to_index["x"]], values[name_to_index["y"]], values[name_to_index["z"]]])
                if has_rgb:
                    rgb_rows.append(
                        [
                            values[name_to_index["red"]],
                            values[name_to_index["green"]],
                            values[name_to_index["blue"]],
                        ]
                    )
    points = torch.tensor(xyz_rows, dtype=torch.float32)
    colors = torch.tensor(rgb_rows, dtype=torch.float32) if has_rgb else None
    return points, colors


def load_colmap_points3d_txt(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    xyz_rows = []
    rgb_rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        xyz_rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb_rows.append([float(parts[4]), float(parts[5]), float(parts[6])])
    if not xyz_rows:
        raise ValueError(f"{path} contains no COLMAP points3D rows.")
    return torch.tensor(xyz_rows, dtype=torch.float32), torch.tensor(rgb_rows, dtype=torch.float32)


def load_colmap_points3d_bin(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    xyz_rows = []
    rgb_rows = []
    with path.open("rb") as fh:
        count_payload = fh.read(8)
        if len(count_payload) != 8:
            raise ValueError(f"{path} is too short for a COLMAP points3D.bin file.")
        (point_count,) = struct.unpack("<Q", count_payload)
        fixed_struct = struct.Struct("<QdddBBBdQ")
        track_struct = struct.Struct("<ii")
        for _ in range(point_count):
            payload = fh.read(fixed_struct.size)
            if len(payload) != fixed_struct.size:
                raise ValueError(f"{path} ended inside a COLMAP point record.")
            values = fixed_struct.unpack(payload)
            _point_id, x, y, z, red, green, blue, _error, track_len = values
            xyz_rows.append([x, y, z])
            rgb_rows.append([float(red), float(green), float(blue)])
            skip = int(track_len) * track_struct.size
            if len(fh.read(skip)) != skip:
                raise ValueError(f"{path} ended inside a COLMAP point track.")
    if not xyz_rows:
        raise ValueError(f"{path} contains no COLMAP points.")
    return torch.tensor(xyz_rows, dtype=torch.float32), torch.tensor(rgb_rows, dtype=torch.float32)


def load_point_cloud_xyz_rgb(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = resolve_point_cloud_path(path)
    if resolved.suffix.lower() == ".ply":
        points, colors = load_ply_point_cloud(resolved)
    elif resolved.name == "points3D.txt":
        points, colors = load_colmap_points3d_txt(resolved)
    elif resolved.name == "points3D.bin":
        points, colors = load_colmap_points3d_bin(resolved)
    else:
        raise ValueError(f"Unsupported point cloud format for {resolved}.")
    finite = torch.isfinite(points).all(dim=-1)
    if colors is not None:
        finite = finite & torch.isfinite(colors).all(dim=-1)
    points = points[finite]
    if points.numel() == 0:
        raise ValueError(f"{resolved} has no finite points.")
    colors = normalize_point_cloud_colors(None if colors is None else colors[finite], int(points.shape[0]))
    return points, colors


def fit_point_cloud_to_powerfoam_box(
    points: torch.Tensor,
    *,
    xy_extent: float,
    z_min: float,
    z_max: float,
) -> torch.Tensor:
    center = points.median(dim=0).values
    centered = points - center
    q95 = torch.quantile(centered.abs(), 0.95, dim=0).clamp_min(1.0e-6)
    xy_scale = 0.85 * float(xy_extent) / torch.max(q95[:2])
    z_scale = 0.45 * (float(z_max) - float(z_min)) / q95[2]
    scale = torch.minimum(xy_scale, z_scale)
    out = centered * scale
    out[:, 2] += 0.5 * (float(z_min) + float(z_max))
    return out


def clamp_point_cloud_to_powerfoam_box(
    points: torch.Tensor,
    *,
    xy_extent: float,
    z_min: float,
    z_max: float,
) -> torch.Tensor:
    out = points.clone()
    out[:, :2] = out[:, :2].clamp(-0.999 * float(xy_extent), 0.999 * float(xy_extent))
    out[:, 2] = out[:, 2].clamp(float(z_min) + 1.0e-4, float(z_max) - 1.0e-4)
    return out


def point_cloud_box_mask(
    points: torch.Tensor,
    *,
    xy_extent: float,
    z_min: float,
    z_max: float,
) -> torch.Tensor:
    return (
        torch.isfinite(points).all(dim=-1)
        & (points[:, 0].abs() <= float(xy_extent))
        & (points[:, 1].abs() <= float(xy_extent))
        & (points[:, 2] >= float(z_min))
        & (points[:, 2] <= float(z_max))
    )


def filter_point_cloud_by_train_visibility(
    points: torch.Tensor,
    colors: torch.Tensor,
    *,
    train_K: torch.Tensor,
    train_w2c: torch.Tensor,
    train_lens_models: list[str] | None = None,
    train_distortions: torch.Tensor | None = None,
    render_size: int,
    min_visible_train_views: int,
    xy_extent: float,
    z_min: float,
    z_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if train_K.ndim != 3:
        raise ValueError(f"train_K must have shape [V,3,3], got {tuple(train_K.shape)}.")
    if train_w2c.ndim == 4:
        train_w2c = train_w2c[:, 0]
    if train_w2c.ndim != 3:
        raise ValueError(f"train_w2c must have shape [V,4,4] or [V,T,4,4], got {tuple(train_w2c.shape)}.")
    if int(train_K.shape[0]) != int(train_w2c.shape[0]):
        raise ValueError(f"train_K/train_w2c view count mismatch: {train_K.shape[0]} vs {train_w2c.shape[0]}.")

    train_K = train_K.detach().to(device=points.device, dtype=points.dtype)
    train_w2c = train_w2c.detach().to(device=points.device, dtype=points.dtype)
    train_distortions = None if train_distortions is None else train_distortions.to(device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=-1)
    visible_votes = torch.zeros(points.shape[0], dtype=torch.int64, device=points.device)
    width = height = int(render_size)
    for view in range(int(train_K.shape[0])):
        points_camera = (points_h @ train_w2c[view].T)[:, :3]
        if train_lens_models is None and train_distortions is None:
            z = points_camera[:, 2]
            u = train_K[view, 0, 0] * points_camera[:, 0] / z.clamp_min(1.0e-6) + train_K[view, 0, 2]
            v = train_K[view, 1, 1] * points_camera[:, 1] / z.clamp_min(1.0e-6) + train_K[view, 1, 2]
            front = z > 1.0e-5
        else:
            camera = CameraSpec(
                fx=train_K[view, 0, 0],
                fy=train_K[view, 1, 1],
                cx=train_K[view, 0, 2],
                cy=train_K[view, 1, 2],
                camera_to_world=torch.linalg.inv(train_w2c[view]),
                lens_model=("pinhole" if train_lens_models is None else train_lens_models[view]),  # type: ignore[arg-type]
                distortion=None if train_distortions is None else train_distortions[view],
            )
            pixels, _depths, _jacobian, front = project_points_camera(points_camera, camera, near_plane=1.0e-5)
            u = pixels[:, 0]
            v = pixels[:, 1]
        inside = front & (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
        visible_votes += inside.to(dtype=torch.int64)

    keep = point_cloud_box_mask(points, xy_extent=xy_extent, z_min=z_min, z_max=z_max) & (
        visible_votes >= int(min_visible_train_views)
    )
    if int(keep.sum().item()) == 0:
        raise ValueError("Point-cloud visibility filtering removed every point.")
    return points[keep].contiguous(), colors[keep].contiguous()


def load_powerfoam_point_cloud_initialization(
    *,
    path: Path,
    frame_count: int,
    cell_count: int,
    xy_extent: float,
    z_min: float,
    z_max: float,
    normalize_mode: str,
    coordinate_frame: str,
    point_transform: torch.Tensor | None = None,
    visibility_filter: str = "none",
    min_visible_train_views: int = 1,
    visibility_train_K: torch.Tensor | None = None,
    visibility_train_w2c: torch.Tensor | None = None,
    visibility_train_lens_models: list[str] | None = None,
    visibility_train_distortions: torch.Tensor | None = None,
    visibility_render_size: int | None = None,
    sample_mode: str = "random",
    duplicate_jitter: float = 0.0,
    seed: int,
) -> PointCloudInitialization:
    resolved = resolve_point_cloud_path(path)
    points, colors = load_point_cloud_xyz_rgb(resolved)
    source_count = int(points.shape[0])
    if point_transform is not None:
        if tuple(point_transform.shape) != (4, 4):
            raise ValueError(f"point_transform must have shape (4, 4), got {tuple(point_transform.shape)}.")
        transform = point_transform.detach().to(device=points.device, dtype=points.dtype)
        points_h = torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype)], dim=-1)
        points = (points_h @ transform.T)[:, :3].contiguous()
    if str(normalize_mode) == "fit_box":
        points = fit_point_cloud_to_powerfoam_box(points, xy_extent=xy_extent, z_min=z_min, z_max=z_max)
    elif str(normalize_mode) != "none":
        raise ValueError("normalize_mode must be 'none' or 'fit_box'")
    if str(visibility_filter) == "train_visible":
        if visibility_train_K is None or visibility_train_w2c is None or visibility_render_size is None:
            raise ValueError("train_visible point-cloud filtering requires train K/w2c camera metadata and render size.")
        points, colors = filter_point_cloud_by_train_visibility(
            points,
            colors,
            train_K=visibility_train_K,
            train_w2c=visibility_train_w2c,
            train_lens_models=visibility_train_lens_models,
            train_distortions=visibility_train_distortions,
            render_size=int(visibility_render_size),
            min_visible_train_views=int(min_visible_train_views),
            xy_extent=xy_extent,
            z_min=z_min,
            z_max=z_max,
        )
    elif str(visibility_filter) != "none":
        raise ValueError("visibility_filter must be 'none' or 'train_visible'")
    filtered_count = int(points.shape[0])
    points = clamp_point_cloud_to_powerfoam_box(points, xy_extent=xy_extent, z_min=z_min, z_max=z_max)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if filtered_count >= int(cell_count):
        if str(sample_mode) == "random":
            sample = torch.randperm(filtered_count, generator=generator)[: int(cell_count)]
        elif str(sample_mode) == "first":
            sample = torch.arange(int(cell_count))
        else:
            raise ValueError("sample_mode must be 'random' or 'first'")
        duplicate_count = 0
    else:
        extra = torch.randint(filtered_count, (int(cell_count) - filtered_count,), generator=generator)
        sample = torch.cat([torch.arange(filtered_count), extra], dim=0)
        duplicate_count = int(extra.numel())
    sampled_points = points.index_select(0, sample).contiguous()
    sampled_colors = colors.index_select(0, sample).contiguous()
    if duplicate_count > 0 and float(duplicate_jitter) > 0.0:
        jitter = float(duplicate_jitter) * torch.randn(
            duplicate_count,
            3,
            generator=generator,
            dtype=sampled_points.dtype,
        )
        sampled_points[filtered_count:] = sampled_points[filtered_count:] + jitter.to(sampled_points.device)
        sampled_points = clamp_point_cloud_to_powerfoam_box(
            sampled_points,
            xy_extent=xy_extent,
            z_min=z_min,
            z_max=z_max,
        )
    return PointCloudInitialization(
        points=sampled_points.unsqueeze(0).repeat(int(frame_count), 1, 1),
        colors=sampled_colors.unsqueeze(0).repeat(int(frame_count), 1, 1),
        source_path=resolved,
        source_count=source_count,
        sampled_count=int(sample.numel()),
        normalize_mode=str(normalize_mode),
        coordinate_frame=str(coordinate_frame),
        visibility_filter=str(visibility_filter),
        sample_mode=str(sample_mode),
        filtered_count=filtered_count,
    )


__all__ = [
    "PointCloudInitialization",
    "clamp_point_cloud_to_powerfoam_box",
    "filter_point_cloud_by_train_visibility",
    "fit_point_cloud_to_powerfoam_box",
    "load_colmap_points3d_bin",
    "load_colmap_points3d_txt",
    "load_ply_point_cloud",
    "load_point_cloud_xyz_rgb",
    "load_powerfoam_point_cloud_initialization",
    "normalize_point_cloud_colors",
    "point_cloud_box_mask",
    "resolve_point_cloud_path",
]
