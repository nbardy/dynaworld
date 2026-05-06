from __future__ import annotations

import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import checkpoint_utils
from checkpoint_utils import atomic_torch_save
from powerfoam_direct import (
    DirectPowerFoamVideo,
    POWERFOAM_SOFTPLUS_BETA,
    PowerFoamRenderOptions,
    camera_facing_quaternion,
    initialize_powerfoam_from_video,
    inverse_softplus,
    quaternion_frames,
    render_powerfoam_torch,
)
from camera import CameraSpec, build_look_at_camera_to_world
from train_powerfoam_direct import (
    LOSS_DEFAULTS,
    flatten_multiview_powerfoam_samples,
    powerfoam_rays_from_camera_grid,
    scheduled_loss_weights,
)


POWERFOAM_UPSTREAM_COMMIT = "96392252ebd0059fe6ca98881b62e12295d9242f"
DIRECT_FIXTURE_PARAM_KEYS = (
    "points",
    "radii",
    "densities",
    "normals",
    "texel_sites",
    "texel_height",
    "texel_sv_axis",
    "texel_sv_rgb",
)
DIRECT_FIXTURE_GRAD_KEYS = (
    ("grad_points", "points"),
    ("grad_radii", "radii"),
    ("grad_density", "densities"),
    ("grad_normals", "normals"),
    ("grad_texel_sites", "texel_sites"),
    ("grad_texel_height", "texel_height"),
    ("grad_texel_sv_axis", "texel_sv_axis"),
    ("grad_texel_sv_rgb", "texel_sv_rgb"),
)
DIRECT_OFFICIAL_CUDA_GRAD_KEYS = (
    # The pinned upstream CUDA/Warp backward prunes tiny texture weights in
    # geometry-sensitive channels; keep this gate on the stable shared channels.
    ("grad_points", "points"),
    ("grad_density", "densities"),
    ("grad_normals", "normals"),
    ("grad_texel_height", "texel_height"),
    ("grad_texel_sv_axis", "texel_sv_axis"),
    ("grad_texel_sv_rgb", "texel_sv_rgb"),
)


def test_powerfoam_metal_save_mp4_uses_quicktime_compatible_h264(tmp_path: Path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe unavailable")
    from video_io import save_mp4

    frames = torch.zeros((2, 3, 16, 16), dtype=torch.float32)
    frames[0, 0].fill_(1.0)
    frames[1, 2].fill_(1.0)
    path = tmp_path / "render.mp4"
    save_mp4(path, frames, fps=2.0)

    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,codec_tag_string,pix_fmt,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    stream = json.loads(probe)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["codec_tag_string"] == "avc1"
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["nb_frames"] == "2"

    frame_path = tmp_path / "frame0.png"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-frames:v",
            "1",
            str(frame_path),
        ],
        check=True,
    )
    from PIL import Image

    frame = torch.from_numpy(np.asarray(Image.open(frame_path).convert("RGB")).copy()).float() / 255.0
    assert float(frame[..., 0].mean()) > 0.9
    assert float(frame[..., 1].mean()) < 0.1
    assert float(frame[..., 2].mean()) < 0.1
    assert float(frame.var()) > 1.0e-2
METAL_FIXTURE_PARAM_KEYS = (
    "points",
    "radii",
    "densities",
    "local_texel_sites",
    "texel_height",
    "texel_sv_axis",
    "texel_sv_rgb",
    "quaternions",
)
METAL_SHARED_GRAD_KEYS = (
    ("grad_density", "densities"),
    ("grad_texel_height", "texel_height"),
    ("grad_texel_sv_axis", "texel_sv_axis"),
    ("grad_texel_sv_rgb", "texel_sv_rgb"),
)


def _load_dynaworld_train_module():
    path = Path(__file__).resolve().parents[1] / "src" / "train" / "train.py"
    spec = importlib.util.spec_from_file_location("dynaworld_train_entry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load train module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_test_metal_model(
    metal_cls,
    make_raster_config,
    render_defaults: dict[str, object],
    *,
    raster_overrides: dict[str, object] | None = None,
    **overrides: object,
):
    kwargs: dict[str, object] = {
        "frame_count": 1,
        "cell_count": 4,
        "render_size": 4,
        "fov_degrees": 55.0,
        "neighbor_count": 2,
        "adjacency_mode": "cech_aabb",
        "xy_extent": 1.0,
        "z_min": 1.0,
        "z_max": 2.0,
        "radius_init": 0.2,
        "radius_min": 0.02,
        "radius_scale": 0.7,
        "density_init": 1.0,
        "feature_mode": "constant",
        "linear_coeff_init": 0.0,
        "linear_coeff_scale": 0.25,
        "normal_init_jitter": 0.0,
        "num_texel_sites": 1,
        "texel_site_scale": 0.5,
        "texel_height_scale": 0.25,
        "sv_dof": 1,
        "sv_axis_init": 1.0,
        "sv_axis_init_jitter": 0.0,
        "sv_rgb_init_jitter": 0.0,
        "color_init_mode": "random",
        "seed": 7,
        "init_frames": None,
        "init_points": None,
        "init_colors": None,
        "image_init_depth": None,
        "image_init_jitter": 0.0,
    }
    kwargs.update(overrides)
    if "raster_config" not in kwargs:
        render_cfg = dict(render_defaults)
        render_cfg["render_size"] = int(kwargs["render_size"])
        if raster_overrides is not None:
            render_cfg.update(raster_overrides)
        kwargs["raster_config"] = make_raster_config(render_cfg)
    return metal_cls(**kwargs)


def _fixture_tensor(payload: dict[str, object], *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.tensor(payload["values"])
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    elif payload.get("dtype") == "float32":
        tensor = tensor.to(dtype=torch.float32)
    elif payload.get("dtype") in {"int32", "int64"}:
        tensor = tensor.to(dtype=torch.long)
    return tensor


def _fixture_param(
    inputs: dict[str, object],
    key: str,
    *,
    device: torch.device | None = None,
    frame: int | None = None,
) -> torch.Tensor:
    tensor = _fixture_tensor(inputs[key], dtype=torch.float32)
    if frame is not None:
        tensor = tensor[frame]
    if device is not None:
        tensor = tensor.to(device)
    return tensor.requires_grad_(True)


def _fixture_grad(params: dict[str, torch.Tensor], key: str, *, frame: int | None = None) -> torch.Tensor:
    grad = params[key].grad
    assert grad is not None, f"{key} grad is missing"
    if frame is not None:
        grad = grad[frame]
    return grad.detach().cpu()


def _fixture_frame_tensors(
    inputs: dict[str, object],
    keys: tuple[str, ...],
    *,
    device: torch.device,
    frame: int = 0,
) -> dict[str, torch.Tensor]:
    return {key: _fixture_tensor(inputs[key], dtype=torch.float32)[frame].to(device) for key in keys}


def _assert_fixture_grads(
    expected: dict[str, object],
    params: dict[str, torch.Tensor],
    grad_keys: tuple[tuple[str, str], ...],
    *,
    atol: float,
    rtol: float,
    frame: int | None = None,
    missing_message: str,
) -> None:
    for expected_key, param_key in grad_keys:
        assert expected_key in expected, f"{missing_message} {expected_key}"
        assert torch.allclose(
            _fixture_grad(params, param_key, frame=frame),
            _fixture_tensor(expected[expected_key], dtype=torch.float32),
            atol=atol,
            rtol=rtol,
        ), expected_key


def _assert_direct_fixture_outputs(
    result: object,
    expected: dict[str, object],
    *,
    atol: float,
    rtol: float = 1.0e-5,
    min_alpha: float | None = None,
) -> None:
    assert torch.allclose(result.rendered, _fixture_tensor(expected["rendered"], dtype=torch.float32), atol=atol, rtol=rtol)
    assert torch.allclose(result.alpha, _fixture_tensor(expected["alpha"], dtype=torch.float32), atol=atol, rtol=rtol)
    if min_alpha is not None:
        assert result.alpha.max() > min_alpha
    assert torch.allclose(
        result.normal_distance,
        _fixture_tensor(expected["normal_distance"], dtype=torch.float32),
        atol=atol,
        rtol=rtol,
    )
    assert torch.allclose(result.contrib, _fixture_tensor(expected["contrib"], dtype=torch.float32), atol=atol, rtol=rtol)
    if "normal" in expected:
        assert torch.allclose(result.normal, _fixture_tensor(expected["normal"], dtype=torch.float32), atol=atol, rtol=rtol)
    assert torch.equal(result.visible_mask.to(dtype=torch.int32), _fixture_tensor(expected["visible_mask"]).to(dtype=torch.int32))


def _render_powerfoam_fixture_with_gradients(
    inputs: dict[str, object],
    options: PowerFoamRenderOptions,
) -> tuple[object, torch.Tensor, dict[str, torch.Tensor]]:
    params = {key: _fixture_param(inputs, key) for key in DIRECT_FIXTURE_PARAM_KEYS}
    result = render_powerfoam_torch(
        params["points"],
        params["radii"],
        params["densities"],
        params["normals"],
        params["texel_sites"],
        None,
        params["texel_height"],
        _fixture_tensor(inputs["adjacency"]).to(dtype=torch.long),
        _fixture_tensor(inputs["rays"], dtype=torch.float32),
        options,
        texel_sv_axis=params["texel_sv_axis"],
        texel_sv_rgb=params["texel_sv_rgb"],
    )
    loss = result.rendered.square().mean() + result.alpha.square().mean() + result.normal_distance.square().mean()
    loss.backward()
    return result, loss, params


def _fixture_csr_adjacency(inputs: dict[str, object], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    padded_adjacency = _fixture_tensor(inputs["adjacency"]).to(dtype=torch.long)[0]
    rows: list[int] = []
    offsets = [0]
    for cell in range(int(padded_adjacency.shape[0])):
        for neighbor in padded_adjacency[cell].tolist():
            if int(neighbor) >= 0:
                rows.append(int(neighbor))
        offsets.append(len(rows))
    return (
        torch.tensor(rows, device=device, dtype=torch.int32),
        torch.tensor(offsets, device=device, dtype=torch.int32),
    )


def _assert_metal_height_sv_fixture_forward_and_shared_backward(
    fixture: dict[str, object],
    *,
    atol: float,
    rtol: float = 1.0e-5,
) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the PowerFoam Metal fixture check")
    try:
        from train_powerfoam_metal import FoamRasterConfig, rasterize_power_foam_quaternion_height_sv_texel_surface
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    inputs = fixture["inputs"]
    expected = fixture["expected"]
    render_options = fixture["render_options"]
    device = torch.device("mps")
    rows_tensor, offsets_tensor = _fixture_csr_adjacency(inputs, device=device)
    config = FoamRasterConfig(
        near_plane=float(render_options["near_plane"]),
        alpha_threshold=float(render_options["alpha_threshold"]),
        transmittance_threshold=float(render_options["transmittance_threshold"]),
        max_alpha=float(render_options["max_alpha"]),
        eps=float(render_options["eps"]),
        texel_temperature=float(render_options["texel_temperature"]),
        use_tiled=True,
    )
    params = {key: _fixture_param(inputs, key, device=device, frame=0) for key in METAL_FIXTURE_PARAM_KEYS}
    rendered, alpha, normal_distance = rasterize_power_foam_quaternion_height_sv_texel_surface(
        params["points"],
        params["radii"],
        params["densities"],
        params["local_texel_sites"],
        params["texel_height"],
        params["texel_sv_axis"],
        params["texel_sv_rgb"],
        params["quaternions"],
        rows_tensor,
        offsets_tensor,
        _fixture_tensor(inputs["rays"], dtype=torch.float32).to(device),
        config,
        return_normal_distance=True,
    )

    expected_rendered = _fixture_tensor(expected["rendered"], dtype=torch.float32)
    assert torch.allclose(rendered.detach().cpu().permute(0, 3, 1, 2), expected_rendered, atol=atol, rtol=rtol)
    assert torch.allclose(alpha.detach().cpu(), _fixture_tensor(expected["alpha"], dtype=torch.float32), atol=atol, rtol=rtol)
    assert torch.allclose(
        normal_distance.detach().cpu(),
        _fixture_tensor(expected["normal_distance"], dtype=torch.float32),
        atol=atol,
        rtol=rtol,
    )

    loss = rendered.square().mean() + alpha.square().mean() + normal_distance.square().mean()
    assert torch.allclose(loss.detach().cpu(), _fixture_tensor(expected["loss"], dtype=torch.float32), atol=atol, rtol=rtol)
    loss.backward()

    # These gradient channels share the same parameterization with the direct
    # Torch/official fixtures. Points, radii, local sites, and quaternions are
    # intentionally not compared here because Metal derives world texel sites
    # and frames from local-site/quaternion parameters, while the fixture stores
    # world-texel-site and normal gradients.
    _assert_fixture_grads(
        expected,
        params,
        METAL_SHARED_GRAD_KEYS,
        atol=atol,
        rtol=rtol,
        missing_message="Metal fixture is missing comparable gradient",
    )


def test_atomic_torch_save_preserves_existing_checkpoint_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "checkpoint.pt"
    target.write_bytes(b"old checkpoint")

    def failing_save(payload: object, path: Path) -> None:
        Path(path).write_bytes(b"partial checkpoint")
        raise RuntimeError("simulated disk-full save failure")

    monkeypatch.setattr(checkpoint_utils.torch, "save", failing_save)
    with pytest.raises(RuntimeError, match="simulated disk-full"):
        atomic_torch_save({"model": {}}, target)

    assert target.read_bytes() == b"old checkpoint"
    assert not (tmp_path / ".checkpoint.pt.tmp").exists()


def test_powerfoam_direct_renderer_backward_smoke() -> None:
    model = DirectPowerFoamVideo(
        frame_count=2,
        cell_count=6,
        render_size=8,
        fov_degrees=55.0,
        neighbor_count=3,
        xy_extent=1.0,
        z_min=1.0,
        z_max=2.0,
        radius_init=0.35,
        radius_min=0.02,
        density_init=1.0,
        seed=3,
        render_options=PowerFoamRenderOptions(),
    )

    rendered, alpha, transmittance, stop_order = model(torch.tensor([0, 1]))
    assert rendered.shape == (2, 3, 8, 8)
    assert alpha.shape == (2, 8, 8)
    assert transmittance.shape == (2, 8, 8)
    assert stop_order.shape == (2, 64)
    result = model(torch.tensor([0, 1]))
    assert result.normal_distance.shape == (2, 8, 8)
    assert result.normal.shape == (2, 3, 8, 8)
    assert result.contrib.shape == (2, 6)
    assert result.point_error.shape == (2, 6)
    assert result.visible_mask.shape == (2, 6)
    assert torch.isfinite(rendered).all()

    loss = rendered.mean() + alpha.mean()
    loss.backward()
    assert model.raw_xy.grad is not None
    assert torch.isfinite(model.raw_xy.grad).all()
    assert model.raw_texel_sv_rgb.grad is not None
    assert torch.isfinite(model.raw_texel_sv_rgb.grad).all()
    assert model.raw_texel_sites.grad is not None
    assert torch.isfinite(model.raw_texel_sites.grad).all()


def test_powerfoam_direct_render_uses_ray_origin_for_geometry() -> None:
    options = PowerFoamRenderOptions(max_alpha=0.99)
    points = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32)
    radii = torch.tensor([[1.0]], dtype=torch.float32)
    densities = torch.tensor([[4.0]], dtype=torch.float32)
    normals = torch.tensor([[[0.0, 0.0, -1.0]]], dtype=torch.float32)
    texel_sites = torch.tensor([[[[0.0, 0.0, 2.0]]]], dtype=torch.float32)
    texel_rgb = torch.tensor([[[[0.7, 0.2, 0.1]]]], dtype=torch.float32)
    texel_height = torch.zeros(1, 1, 1, dtype=torch.float32)
    adjacency = torch.empty(1, 1, 0, dtype=torch.long)
    rays = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)

    base = render_powerfoam_torch(
        points,
        radii,
        densities,
        normals,
        texel_sites,
        texel_rgb,
        texel_height,
        adjacency,
        rays,
        options,
    )

    shift = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32)
    translated_rays = rays.clone()
    translated_rays[..., :3] += shift
    translated = render_powerfoam_torch(
        points + shift,
        radii,
        densities,
        normals,
        texel_sites + shift,
        texel_rgb,
        texel_height,
        adjacency,
        translated_rays,
        options,
    )

    assert torch.allclose(translated.rendered, base.rendered, atol=1.0e-6)
    assert torch.allclose(translated.alpha, base.alpha, atol=1.0e-6)


def test_powerfoam_direct_sv_color_uses_ray_origin() -> None:
    options = PowerFoamRenderOptions(max_alpha=0.99)
    points = torch.tensor([[[0.0, 0.0, 2.0]], [[0.0, 0.0, 2.0]]], dtype=torch.float32)
    radii = torch.ones(2, 1, dtype=torch.float32)
    densities = torch.full((2, 1), 4.0, dtype=torch.float32)
    normals = torch.tensor([[[0.0, 0.0, -1.0]], [[0.0, 0.0, -1.0]]], dtype=torch.float32)
    texel_sites = torch.tensor([[[[0.0, 0.0, 2.0]]], [[[0.0, 0.0, 2.0]]]], dtype=torch.float32)
    texel_height = torch.zeros(2, 1, 1, dtype=torch.float32)
    adjacency = torch.empty(2, 1, 0, dtype=torch.long)
    axis_front = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    axis_side = torch.nn.functional.normalize(torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32), dim=0)
    axes = 24.0 * torch.stack([axis_front, axis_side], dim=0).view(1, 1, 1, 2, 3).repeat(2, 1, 1, 1, 1)
    colors = torch.tensor(
        [[[[[0.45, -0.45, -0.45], [-0.45, 0.45, -0.45]]]]],
        dtype=torch.float32,
    ).repeat(2, 1, 1, 1, 1)
    ray0 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    dir1 = torch.nn.functional.normalize(torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32), dim=0)
    ray1 = torch.cat([torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32), dir1], dim=0)
    rays = torch.stack([ray0, ray1], dim=0).view(2, 1, 1, 6)

    result = render_powerfoam_torch(
        points,
        radii,
        densities,
        normals,
        texel_sites,
        None,
        texel_height,
        adjacency,
        rays,
        options,
        texel_sv_axis=axes,
        texel_sv_rgb=colors,
    )

    front_rgb = result.rendered[0, :, 0, 0]
    side_rgb = result.rendered[1, :, 0, 0]
    assert front_rgb[0] > front_rgb[1]
    assert side_rgb[1] > side_rgb[0]


def test_powerfoam_direct_shared_state_accepts_posed_multiview_rays() -> None:
    device = torch.device("cpu")
    image_size = 4
    cameras = (
        (
            CameraSpec(
                fx=4.0,
                fy=4.0,
                cx=2.0,
                cy=2.0,
                camera_to_world=build_look_at_camera_to_world(torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)),
            ),
        ),
        (
            CameraSpec(
                fx=4.0,
                fy=4.0,
                cx=2.0,
                cy=2.0,
                camera_to_world=build_look_at_camera_to_world(torch.tensor([0.35, 0.0, -1.0], dtype=torch.float32)),
            ),
        ),
    )
    rays = powerfoam_rays_from_camera_grid(cameras, height=image_size, width=image_size, device=device)
    frames = torch.zeros(2, 1, 3, image_size, image_size, dtype=torch.float32)
    frames[0, 0, 0] = 0.25
    frames[1, 0, 1] = 0.25
    targets, frame_indices, sample_rays = flatten_multiview_powerfoam_samples(frames, rays)
    assert frame_indices.tolist() == [0, 0]
    assert sample_rays.shape == (2, image_size, image_size, 6)
    assert not torch.allclose(sample_rays[0, ..., :3], sample_rays[1, ..., :3])

    model = DirectPowerFoamVideo(
        frame_count=1,
        cell_count=5,
        render_size=image_size,
        fov_degrees=55.0,
        neighbor_count=4,
        xy_extent=0.5,
        z_min=1.0,
        z_max=2.0,
        radius_init=0.5,
        radius_min=0.02,
        density_init=2.0,
        seed=4,
        render_options=PowerFoamRenderOptions(),
    )
    result = model(frame_indices, target_rgb=targets, rays=sample_rays)
    assert result.rendered.shape == (2, 3, image_size, image_size)
    loss = F.l1_loss(result.rendered, targets) + result.alpha.mean()
    loss.backward()
    assert model.raw_xy.grad is not None
    assert model.raw_xy.grad.shape[0] == 1
    assert torch.isfinite(model.raw_xy.grad).all()


def test_powerfoam_direct_loads_canonical_origin_parity_fixture() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "research_experiments"
        / "dynamic_foam"
        / "fixtures"
        / "powerfoam_tiny_height_sv_origin_parity_v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["metadata"]["upstream_powerfoam_commit"] == POWERFOAM_UPSTREAM_COMMIT
    inputs = fixture["inputs"]
    expected = fixture["expected"]
    options = PowerFoamRenderOptions(**fixture["render_options"])

    result = render_powerfoam_torch(
        _fixture_tensor(inputs["points"], dtype=torch.float32),
        _fixture_tensor(inputs["radii"], dtype=torch.float32),
        _fixture_tensor(inputs["densities"], dtype=torch.float32),
        _fixture_tensor(inputs["normals"], dtype=torch.float32),
        _fixture_tensor(inputs["texel_sites"], dtype=torch.float32),
        None,
        _fixture_tensor(inputs["texel_height"], dtype=torch.float32),
        _fixture_tensor(inputs["adjacency"]).to(dtype=torch.long),
        _fixture_tensor(inputs["rays"], dtype=torch.float32),
        options,
        texel_sv_axis=_fixture_tensor(inputs["texel_sv_axis"], dtype=torch.float32),
        texel_sv_rgb=_fixture_tensor(inputs["texel_sv_rgb"], dtype=torch.float32),
    )

    _assert_direct_fixture_outputs(result, expected, atol=1.0e-7)


def test_powerfoam_direct_loads_official_camera_local_fixture() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "research_experiments"
        / "dynamic_foam"
        / "fixtures"
        / "powerfoam_tiny_height_sv_official_camera_local_v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["metadata"]["backend"] == "local"
    assert fixture["metadata"]["upstream_powerfoam_commit"] == POWERFOAM_UPSTREAM_COMMIT
    inputs = fixture["inputs"]
    expected = fixture["expected"]
    options = PowerFoamRenderOptions(**fixture["render_options"])

    result, loss, params = _render_powerfoam_fixture_with_gradients(inputs, options)

    _assert_direct_fixture_outputs(result, expected, atol=1.0e-7, min_alpha=0.4)
    assert torch.allclose(loss.detach(), _fixture_tensor(expected["loss"], dtype=torch.float32), atol=1.0e-7)
    _assert_fixture_grads(
        expected,
        params,
        DIRECT_FIXTURE_GRAD_KEYS,
        atol=1.0e-7,
        rtol=1.0e-7,
        frame=0,
        missing_message="local official-camera fixture is missing",
    )


def test_powerfoam_direct_matches_official_cuda_fixture_if_present() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "research_experiments"
        / "dynamic_foam"
        / "fixtures"
        / "powerfoam_tiny_height_sv_official_camera_official_v1.json"
    )
    if not fixture_path.exists():
        pytest.skip("official CUDA/Warp PowerFoam fixture has not been generated on a CUDA host")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["metadata"]["backend"] == "official"
    assert fixture["metadata"]["upstream_powerfoam_commit"] == POWERFOAM_UPSTREAM_COMMIT
    inputs = fixture["inputs"]
    expected = fixture["expected"]
    options = PowerFoamRenderOptions(**fixture["render_options"])

    result, loss, params = _render_powerfoam_fixture_with_gradients(inputs, options)

    _assert_direct_fixture_outputs(result, expected, atol=1.0e-4, rtol=1.0e-3)
    assert torch.allclose(loss.detach(), _fixture_tensor(expected["loss"], dtype=torch.float32), atol=1.0e-4)
    _assert_fixture_grads(
        expected,
        params,
        DIRECT_OFFICIAL_CUDA_GRAD_KEYS,
        atol=1.0e-4,
        rtol=1.0e-3,
        frame=0,
        missing_message="official fixture is missing",
    )


def test_powerfoam_metal_loads_canonical_origin_parity_fixture() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required for the PowerFoam Metal fixture check")
    try:
        from train_powerfoam_metal import FoamRasterConfig, rasterize_power_foam_quaternion_height_sv_texel_surface
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "research_experiments"
        / "dynamic_foam"
        / "fixtures"
        / "powerfoam_tiny_height_sv_origin_parity_v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    inputs = fixture["inputs"]
    expected = fixture["expected"]
    render_options = fixture["render_options"]
    device = torch.device("mps")

    rows_tensor, offsets_tensor = _fixture_csr_adjacency(inputs, device=device)
    config = FoamRasterConfig(
        near_plane=float(render_options["near_plane"]),
        alpha_threshold=float(render_options["alpha_threshold"]),
        transmittance_threshold=float(render_options["transmittance_threshold"]),
        max_alpha=float(render_options["max_alpha"]),
        eps=float(render_options["eps"]),
        texel_temperature=float(render_options["texel_temperature"]),
        use_tiled=True,
    )
    params = _fixture_frame_tensors(inputs, METAL_FIXTURE_PARAM_KEYS, device=device)

    rendered, alpha, normal_distance = rasterize_power_foam_quaternion_height_sv_texel_surface(
        params["points"],
        params["radii"],
        params["densities"],
        params["local_texel_sites"],
        params["texel_height"],
        params["texel_sv_axis"],
        params["texel_sv_rgb"],
        params["quaternions"],
        rows_tensor,
        offsets_tensor,
        _fixture_tensor(inputs["rays"], dtype=torch.float32).to(device),
        config,
        return_normal_distance=True,
    )

    expected_rendered = _fixture_tensor(expected["rendered"], dtype=torch.float32)
    assert torch.allclose(rendered.detach().cpu().permute(0, 3, 1, 2), expected_rendered, atol=1.0e-5)
    assert torch.allclose(alpha.detach().cpu(), _fixture_tensor(expected["alpha"], dtype=torch.float32), atol=1.0e-5)
    assert torch.allclose(
        normal_distance.detach().cpu(),
        _fixture_tensor(expected["normal_distance"], dtype=torch.float32),
        atol=1.0e-5,
    )


def test_powerfoam_metal_matches_official_camera_local_fixture_shared_backward() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "research_experiments"
        / "dynamic_foam"
        / "fixtures"
        / "powerfoam_tiny_height_sv_official_camera_local_v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["metadata"]["backend"] == "local"
    assert fixture["metadata"]["upstream_powerfoam_commit"] == POWERFOAM_UPSTREAM_COMMIT
    _assert_metal_height_sv_fixture_forward_and_shared_backward(fixture, atol=1.0e-5)


def test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "research_experiments"
        / "dynamic_foam"
        / "fixtures"
        / "powerfoam_tiny_height_sv_official_camera_official_v1.json"
    )
    if not fixture_path.exists():
        pytest.skip("official CUDA/Warp PowerFoam fixture has not been generated on a CUDA host")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["metadata"]["backend"] == "official"
    assert fixture["metadata"]["upstream_powerfoam_commit"] == POWERFOAM_UPSTREAM_COMMIT
    _assert_metal_height_sv_fixture_forward_and_shared_backward(fixture, atol=1.0e-4, rtol=1.0e-3)


def test_powerfoam_texel_sites_are_radius_scaled_and_sv_color_detaches_geometry() -> None:
    model = DirectPowerFoamVideo(
        frame_count=1,
        cell_count=1,
        render_size=4,
        fov_degrees=55.0,
        neighbor_count=0,
        xy_extent=1.0,
        z_min=1.0,
        z_max=2.0,
        radius_init=0.25,
        radius_min=0.0,
        density_init=1.0,
        seed=5,
        render_options=PowerFoamRenderOptions(),
    )
    with torch.no_grad():
        model.raw_xy.zero_()
        model.raw_z.zero_()
        model.raw_quaternions.copy_(torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))
        model.raw_texel_sites.fill_(0.0)
        model.raw_texel_sites[..., 0] = 0.5
        model.raw_radii.fill_(0.25)

    points, radii, _, normals, texel_sites, texel_rgb, _ = model.decoded_powerfoam_parameters()
    offset = texel_sites - points[:, :, None, :]
    tangent_offset = offset[..., 1]
    assert torch.allclose(tangent_offset, 0.5 * radii[:, :, None], atol=1.0e-5)
    assert torch.allclose(normals[..., 0], torch.ones_like(normals[..., 0]), atol=1.0e-5)

    model.zero_grad(set_to_none=True)
    texel_rgb.sum().backward()
    assert model.raw_texel_sites.grad is None


def test_powerfoam_direct_config_dispatches_to_trainer() -> None:
    trainer_entry_for_config = _load_dynaworld_train_module().trainer_entry_for_config
    entry = trainer_entry_for_config("src/train_configs/local_mac_powerfoam_direct_128_smoke.jsonc")
    assert entry.module == "train_powerfoam_direct"
    metal_entry = trainer_entry_for_config("src/train_configs/local_mac_powerfoam_metal_video_64_smoke.jsonc")
    assert metal_entry.module == "train_powerfoam_metal"
    metal_256_entry = trainer_entry_for_config("src/train_configs/local_mac_powerfoam_metal_video_256_smoke.jsonc")
    assert metal_256_entry.module == "train_powerfoam_metal"
    metal_1024_entry = trainer_entry_for_config("src/train_configs/local_mac_powerfoam_metal_video_1024_smoke.jsonc")
    assert metal_1024_entry.module == "train_powerfoam_metal"
    metal_linear_entry = trainer_entry_for_config("src/train_configs/local_mac_powerfoam_metal_linear_video_1024_smoke.jsonc")
    assert metal_linear_entry.module == "train_powerfoam_metal"
    metal_surface_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_powerfoam_metal_surface_linear_video_1024_smoke.jsonc"
    )
    assert metal_surface_entry.module == "train_powerfoam_metal"
    metal_oriented_surface_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_powerfoam_metal_oriented_surface_linear_video_1024_smoke.jsonc"
    )
    assert metal_oriented_surface_entry.module == "train_powerfoam_metal"
    metal_texel_surface_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_powerfoam_metal_oriented_texel_surface_video_1024_smoke.jsonc"
    )
    assert metal_texel_surface_entry.module == "train_powerfoam_metal"
    metal_texel_random_color_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_powerfoam_metal_oriented_texel_surface_random_color_video_1024_smoke.jsonc"
    )
    assert metal_texel_random_color_entry.module == "train_powerfoam_metal"
    metal_quaternion_texel_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_powerfoam_metal_quaternion_texel_surface_video_1024_smoke.jsonc"
    )
    assert metal_quaternion_texel_entry.module == "train_powerfoam_metal"
    metal_quaternion_height_texel_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_powerfoam_metal_quaternion_height_texel_surface_video_1024_smoke.jsonc"
    )
    assert metal_quaternion_height_texel_entry.module == "train_powerfoam_metal"
    metal_quaternion_height_sv_texel_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_powerfoam_metal_quaternion_height_sv_texel_surface_video_1024_smoke.jsonc"
    )
    assert metal_quaternion_height_sv_texel_entry.module == "train_powerfoam_metal"
    metal_quaternion_height_sv_texel_tiled_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_powerfoam_metal_quaternion_height_sv_texel_surface_tiled_video_1024_smoke.jsonc"
    )
    assert metal_quaternion_height_sv_texel_tiled_entry.module == "train_powerfoam_metal"
    metal_point_cloud_init_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_powerfoam_metal_point_cloud_init_quaternion_height_sv_tiled_32_smoke.jsonc"
    )
    assert metal_point_cloud_init_entry.module == "train_powerfoam_metal"
    metal_official_lr_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_powerfoam_metal_official_lr_schedule_quaternion_height_sv_tiled_32_smoke.jsonc"
    )
    assert metal_official_lr_entry.module == "train_powerfoam_metal"
    metal_multicam_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_tiled_32_smoke.jsonc"
    )
    assert metal_multicam_entry.module == "train_powerfoam_metal"
    dynamic_gauge_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_dynamic_gauge_foam_video_1024_smoke.jsonc"
    )
    assert dynamic_gauge_entry.module == "train_dynamic_gauge_foam"
    dynamic_powerfoam_smooth_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_dynamic_powerfoam_metal_per_frame_smooth_1024_smoke.jsonc"
    )
    assert dynamic_powerfoam_smooth_entry.module == "train_dynamic_powerfoam_metal"
    dynamic_powerfoam_rbf_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_dynamic_powerfoam_metal_rbf_1024_smoke.jsonc"
    )
    assert dynamic_powerfoam_rbf_entry.module == "train_dynamic_powerfoam_metal"
    token_dynamic_powerfoam_feature_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_token_dynamic_powerfoam_features_F16_1024_smoke.jsonc"
    )
    assert token_dynamic_powerfoam_feature_entry.module == "train_dynamic_powerfoam_metal"
    token_dynamic_powerfoam_feature_f32_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_smoke.jsonc"
    )
    assert token_dynamic_powerfoam_feature_f32_entry.module == "train_dynamic_powerfoam_metal"
    token_dynamic_powerfoam_motion_probe_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_motion_probe.jsonc"
    )
    assert token_dynamic_powerfoam_motion_probe_entry.module == "train_dynamic_powerfoam_metal"
    token_dynamic_powerfoam_512px_entry = trainer_entry_for_config(
        "src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_512px_smoke.jsonc"
    )
    assert token_dynamic_powerfoam_512px_entry.module == "train_dynamic_powerfoam_metal"


def test_powerfoam_metal_camera_rays_include_camera_pose() -> None:
    try:
        from camera import CameraSpec
        from train_powerfoam_metal import powerfoam_rays_from_camera
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    camera = CameraSpec(
        fx=4.0,
        fy=4.0,
        cx=2.0,
        cy=2.0,
        camera_to_world=c2w,
    )

    rays = powerfoam_rays_from_camera(camera, height=4, width=4, device=torch.device("cpu"))

    assert rays.shape == (1, 4, 4, 6)
    assert torch.allclose(rays[0, ..., :3], c2w[:3, 3].view(1, 1, 3).expand(4, 4, 3))
    assert torch.allclose(rays[0, ..., 3:].norm(dim=-1), torch.ones(4, 4), atol=1.0e-6)


def test_powerfoam_metal_multiview_flatten_shares_frame_indices_across_views() -> None:
    try:
        from train_powerfoam_metal import flatten_multiview_powerfoam_samples
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    frames = torch.arange(2 * 3 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 3, 2, 2)
    rays = torch.zeros(2, 3, 2, 2, 6, dtype=torch.float32)

    targets, frame_indices, sample_rays = flatten_multiview_powerfoam_samples(frames, rays)

    assert targets.shape == (6, 3, 2, 2)
    assert sample_rays.shape == (6, 2, 2, 6)
    assert frame_indices.tolist() == [0, 1, 2, 0, 1, 2]
    assert torch.equal(targets[3], frames[1, 0])


def test_powerfoam_metal_ssim_loss_is_zero_for_identical_images() -> None:
    try:
        from train_powerfoam_metal import LOSS_DEFAULTS, powerfoam_ssim_loss
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    image = torch.rand(2, 3, 6, 6)
    loss_cfg = dict(LOSS_DEFAULTS)
    loss_cfg["ssim_window_size"] = 11

    loss = powerfoam_ssim_loss(image, image, loss_cfg)

    assert float(loss) == pytest.approx(0.0, abs=1.0e-6)


def test_powerfoam_metal_background_compositing_uses_alpha() -> None:
    try:
        from train_powerfoam_metal import composite_powerfoam_background, training_background_tensor
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    rendered = torch.zeros(2, 3, 2, 2)
    alpha = torch.tensor(
        [
            [[0.0, 0.5], [1.0, 0.25]],
            [[0.75, 1.0], [0.0, 0.5]],
        ],
        dtype=torch.float32,
    )
    background = torch.tensor([[[[0.2]], [[0.4]], [[0.8]]]], dtype=torch.float32)
    composited = composite_powerfoam_background(rendered, alpha, background)

    assert torch.allclose(composited[:, 0], (1.0 - alpha) * 0.2)
    random_bg = training_background_tensor(
        rendered,
        {"background_mode": "random", "background": [0.0, 0.0, 0.0]},
    )
    assert random_bg.shape == (2, 3, 1, 1)
    assert torch.all((random_bg >= 0.0) & (random_bg <= 1.0))


def test_powerfoam_metal_point_cloud_init_loads_ply_static_geometry(tmp_path: Path) -> None:
    try:
        from train_powerfoam_metal import (
            MetalPowerFoamVideo,
            RENDER_DEFAULTS,
            load_powerfoam_point_cloud_initialization,
            make_raster_config,
        )
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    ply_path = tmp_path / "input.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "-0.20 0.10 1.40 255 0 0",
                "0.25 -0.15 1.60 0 255 0",
                "0.05 0.20 1.80 0 0 255",
                "-0.10 -0.05 2.00 128 64 32",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    init = load_powerfoam_point_cloud_initialization(
        path=ply_path,
        frame_count=2,
        cell_count=3,
        xy_extent=1.0,
        z_min=1.0,
        z_max=2.2,
        normalize_mode="none",
        coordinate_frame="model",
        seed=3,
    )
    model = MetalPowerFoamVideo(
        frame_count=2,
        cell_count=3,
        render_size=4,
        fov_degrees=55.0,
        neighbor_count=2,
        adjacency_mode="cech_aabb",
        xy_extent=1.0,
        z_min=1.0,
        z_max=2.2,
        radius_init=0.2,
        radius_min=0.02,
        radius_scale=0.7,
        density_init=1.0,
        feature_mode="quaternion_height_sv_texel_surface",
        linear_coeff_init=0.0,
        linear_coeff_scale=0.25,
        normal_init_jitter=0.0,
        num_texel_sites=2,
        texel_site_scale=0.5,
        texel_height_scale=0.25,
        sv_dof=2,
        sv_axis_init=1.0,
        sv_axis_init_jitter=0.0,
        sv_rgb_init_jitter=0.0,
        color_init_mode="image",
        seed=5,
        init_frames=None,
        init_points=init.points,
        init_colors=init.colors,
        image_init_depth=None,
        image_init_jitter=0.0,
        raster_config=make_raster_config(RENDER_DEFAULTS),
    )

    points, radii, densities, features, normals = model.decoded_parameters()

    assert init.source_count == 4
    assert init.sampled_count == 3
    assert torch.allclose(points, init.points, atol=1.0e-5)
    assert torch.allclose(features.mean(dim=2), init.colors, atol=1.0e-6)
    assert torch.isfinite(radii).all()
    assert torch.isfinite(densities).all()
    assert normals is not None and torch.isfinite(normals).all()


def test_powerfoam_point_cloud_init_applies_world_to_model_transform(tmp_path: Path) -> None:
    try:
        from train_powerfoam_metal import load_powerfoam_point_cloud_initialization
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    ply_path = tmp_path / "world.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "11.0 22.0 33.0 10 20 30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    world_to_model = torch.eye(4, dtype=torch.float32)
    world_to_model[:3, 3] = torch.tensor([-10.0, -20.0, -30.0], dtype=torch.float32)
    init = load_powerfoam_point_cloud_initialization(
        path=ply_path,
        frame_count=1,
        cell_count=1,
        xy_extent=4.0,
        z_min=1.0,
        z_max=5.0,
        normalize_mode="none",
        coordinate_frame="multicam_world",
        point_transform=world_to_model,
        seed=0,
    )
    torch.testing.assert_close(init.points[0, 0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(init.colors[0, 0], torch.tensor([10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0]))
    assert init.coordinate_frame == "multicam_world"
    assert init.visibility_filter == "none"


def test_powerfoam_point_cloud_init_can_keep_ply_order_for_ranked_points(tmp_path: Path) -> None:
    try:
        from train_powerfoam_metal import load_powerfoam_point_cloud_initialization
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    ply_path = tmp_path / "ranked.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0.0 0.0 1.1 10 0 0",
                "0.1 0.0 1.2 20 0 0",
                "0.2 0.0 1.3 30 0 0",
                "0.3 0.0 1.4 40 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    init = load_powerfoam_point_cloud_initialization(
        path=ply_path,
        frame_count=1,
        cell_count=2,
        xy_extent=1.0,
        z_min=1.0,
        z_max=2.0,
        normalize_mode="none",
        coordinate_frame="model",
        sample_mode="first",
        seed=123,
    )
    assert init.sample_mode == "first"
    torch.testing.assert_close(init.points[0], torch.tensor([[0.0, 0.0, 1.1], [0.1, 0.0, 1.2]]))
    torch.testing.assert_close(init.colors[0, :, 0], torch.tensor([10.0 / 255.0, 20.0 / 255.0]))


def test_powerfoam_point_cloud_init_filters_train_visible_points(tmp_path: Path) -> None:
    try:
        from train_powerfoam_metal import load_powerfoam_point_cloud_initialization
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    ply_path = tmp_path / "points.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0.0 0.0 2.0 255 0 0",
                "10.0 0.0 2.0 0 255 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_K = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    train_w2c = torch.eye(4, dtype=torch.float32).view(1, 4, 4)
    init = load_powerfoam_point_cloud_initialization(
        path=ply_path,
        frame_count=1,
        cell_count=1,
        xy_extent=12.0,
        z_min=1.0,
        z_max=3.0,
        normalize_mode="none",
        coordinate_frame="model",
        visibility_filter="train_visible",
        min_visible_train_views=1,
        visibility_train_K=train_K,
        visibility_train_w2c=train_w2c,
        visibility_render_size=3,
        seed=0,
    )
    assert init.source_count == 2
    assert init.filtered_count == 1
    assert init.visibility_filter == "train_visible"
    torch.testing.assert_close(init.points[0, 0], torch.tensor([0.0, 0.0, 2.0]))
    torch.testing.assert_close(init.colors[0, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_powerfoam_point_cloud_init_jitters_duplicate_backfill(tmp_path: Path) -> None:
    try:
        from train_powerfoam_metal import load_powerfoam_point_cloud_initialization
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    ply_path = tmp_path / "single_point.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0.0 0.0 2.0 255 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    init = load_powerfoam_point_cloud_initialization(
        path=ply_path,
        frame_count=1,
        cell_count=4,
        xy_extent=1.0,
        z_min=1.0,
        z_max=3.0,
        normalize_mode="none",
        coordinate_frame="model",
        visibility_filter="none",
        duplicate_jitter=0.05,
        seed=123,
    )
    assert init.source_count == 1
    assert init.filtered_count == 1
    assert init.sampled_count == 4
    torch.testing.assert_close(init.points[0, 0], torch.tensor([0.0, 0.0, 2.0]))
    assert bool(torch.linalg.vector_norm(init.points[0, 1:] - init.points[0, :1], dim=-1).min() > 1.0e-6)
    assert bool((init.points[0, :, :2].abs() <= 1.0).all())
    assert bool(((init.points[0, :, 2] >= 1.0) & (init.points[0, :, 2] <= 3.0)).all())


def test_powerfoam_metal_synthetic_posed_views_overfit_shared_state() -> None:
    try:
        from camera import CameraSpec, build_look_at_camera_to_world
        from train_powerfoam_metal import (
            MetalPowerFoamVideo,
            RENDER_DEFAULTS,
            TRAIN_DEFAULTS,
            make_raster_config,
            powerfoam_rays_from_camera,
        )
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")
    if not torch.backends.mps.is_available():
        pytest.skip("PowerFoam Metal synthetic overfit requires MPS.")

    device = torch.device("mps")
    render_size = 16
    render_cfg = dict(RENDER_DEFAULTS)
    render_cfg.update({"render_size": render_size, "use_tiled": True, "tiled_builder": "auto"})
    raster_config = make_raster_config(render_cfg)

    def camera_at(position: list[float]) -> CameraSpec:
        c2w = build_look_at_camera_to_world(
            torch.tensor(position, dtype=torch.float32),
            target=torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32),
        )
        return CameraSpec(
            fx=18.0,
            fy=18.0,
            cx=float(render_size) / 2.0,
            cy=float(render_size) / 2.0,
            camera_to_world=c2w,
        )

    rays = torch.cat(
        [
            powerfoam_rays_from_camera(camera, height=render_size, width=render_size, device=device)
            for camera in (
                camera_at([0.0, 0.0, -1.0]),
                camera_at([0.4, 0.0, -1.0]),
                camera_at([-0.4, 0.0, -1.0]),
            )
        ],
        dim=0,
    )
    frame_indices = torch.zeros(rays.shape[0], dtype=torch.long, device=device)
    common_kwargs = dict(
        frame_count=1,
        render_size=render_size,
        fov_degrees=55.0,
        adjacency_mode="cech_aabb",
        xy_extent=1.5,
        z_min=0.5,
        z_max=3.0,
        radius_min=0.02,
        radius_scale=0.72,
        feature_mode="constant",
        linear_coeff_init=0.0,
        linear_coeff_scale=0.25,
        normal_init_jitter=0.0,
        num_texel_sites=1,
        texel_site_scale=0.5,
        texel_height_scale=0.25,
        sv_dof=1,
        sv_axis_init=1.0,
        sv_axis_init_jitter=0.0,
        sv_rgb_init_jitter=0.0,
        color_init_mode="random",
        init_frames=None,
        init_points=None,
        init_colors=None,
        image_init_depth=2.0,
        image_init_jitter=0.1,
        raster_config=raster_config,
    )

    teacher = MetalPowerFoamVideo(
        cell_count=1,
        neighbor_count=0,
        radius_init=0.45,
        density_init=8.0,
        seed=1,
        **common_kwargs,
    ).to(device)
    with torch.no_grad():
        teacher.raw_xy.zero_()
        teacher.raw_z.zero_()
        teacher.raw_radii.fill_(0.35)
        teacher.raw_densities.fill_(3.0)
        teacher.raw_features[0, 0] = torch.tensor([1.0, -0.3, -0.8], device=device)
        target, _alpha = teacher(frame_indices, rays=rays)

    student = MetalPowerFoamVideo(
        cell_count=4,
        neighbor_count=3,
        radius_init=0.3,
        density_init=2.0,
        seed=4,
        **common_kwargs,
    ).to(device)
    train_cfg = dict(TRAIN_DEFAULTS)
    train_cfg.update(
        {
            "lr": 0.05,
            "point_lr_multiplier": 0.5,
            "radius_lr_multiplier": 0.2,
            "density_lr_multiplier": 0.2,
        }
    )
    optimizer = torch.optim.Adam(student.optimizer_param_groups(train_cfg), lr=float(train_cfg["lr"]))

    first_l1 = None
    l1 = None
    for _step in range(61):
        rendered, _alpha = student(frame_indices, rays=rays)
        l1 = (rendered - target).abs().mean()
        if first_l1 is None:
            first_l1 = float(l1.detach().cpu())
        loss = l1 + 0.1 * (rendered - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert first_l1 is not None and l1 is not None
    final_l1 = float(l1.detach().cpu())
    assert final_l1 < 0.006
    assert final_l1 < first_l1 * 0.25
    assert student.parameter_drift_metrics()["state_mean_center_delta"] > 0.0


def test_powerfoam_metal_height_sv_raytrace_overfits_tiny_material() -> None:
    try:
        from camera import CameraSpec, build_look_at_camera_to_world
        from train_powerfoam_metal import (
            MetalPowerFoamVideo,
            RENDER_DEFAULTS,
            TRAIN_DEFAULTS,
            make_raster_config,
            powerfoam_rays_from_camera,
        )
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")
    if not torch.backends.mps.is_available():
        pytest.skip("PowerFoam Metal height+SV raytrace overfit requires MPS.")

    device = torch.device("mps")
    render_size = 12
    render_cfg = dict(RENDER_DEFAULTS)
    render_cfg.update({"render_size": render_size, "use_raytrace": True})
    raster_config = make_raster_config(render_cfg)

    def camera_at(position: list[float]) -> CameraSpec:
        c2w = build_look_at_camera_to_world(
            torch.tensor(position, dtype=torch.float32),
            target=torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32),
        )
        return CameraSpec(
            fx=16.0,
            fy=16.0,
            cx=float(render_size) / 2.0,
            cy=float(render_size) / 2.0,
            camera_to_world=c2w,
        )

    rays = torch.cat(
        [
            powerfoam_rays_from_camera(camera, height=render_size, width=render_size, device=device)
            for camera in (camera_at([0.0, 0.0, -1.0]), camera_at([0.25, 0.0, -1.0]))
        ],
        dim=0,
    )
    frame_indices = torch.zeros(rays.shape[0], dtype=torch.long, device=device)
    common_kwargs = dict(
        frame_count=1,
        cell_count=1,
        render_size=render_size,
        fov_degrees=55.0,
        neighbor_count=0,
        adjacency_mode="cech_aabb",
        xy_extent=1.5,
        z_min=0.5,
        z_max=3.0,
        radius_init=0.5,
        radius_min=0.02,
        radius_scale=0.72,
        density_init=8.0,
        feature_mode="quaternion_height_sv_texel_surface",
        linear_coeff_init=0.0,
        linear_coeff_scale=0.25,
        normal_init_jitter=0.0,
        num_texel_sites=3,
        texel_site_scale=0.5,
        texel_height_scale=0.1,
        sv_dof=1,
        sv_axis_init=1.0,
        sv_axis_init_jitter=0.0,
        sv_rgb_init_jitter=0.0,
        color_init_mode="image",
        init_frames=None,
        init_points=None,
        init_colors=None,
        image_init_depth=2.0,
        image_init_jitter=0.0,
        raster_config=raster_config,
        use_raytrace=True,
    )

    def set_shared_geometry(model: torch.nn.Module) -> None:
        model.raw_xy.zero_()
        model.raw_z.fill_(math.log(0.6 / 0.4))
        model.raw_radii.fill_(0.2)
        model.raw_densities.fill_(3.0)
        model.raw_texel_sites.zero_()
        model.raw_texel_heights.zero_()
        model.raw_texel_sv_axis.zero_()
        model.raw_texel_sv_axis[..., 2] = 1.0

    teacher = MetalPowerFoamVideo(seed=1, **common_kwargs).to(device)
    student = MetalPowerFoamVideo(seed=2, **common_kwargs).to(device)
    with torch.no_grad():
        set_shared_geometry(teacher)
        set_shared_geometry(student)
        teacher.raw_texel_sv_rgb[:] = torch.tensor([0.35, -0.05, -0.3], device=device)
        student.raw_texel_sv_rgb.fill_(-0.35)
        target, _alpha = teacher(frame_indices, rays=rays)

    train_cfg = dict(TRAIN_DEFAULTS)
    train_cfg.update(
        {
            "points_lr_init": 0.0,
            "radii_lr_init": 0.0,
            "density_lr_init": 0.0,
            "quaternions_lr_init": 0.0,
            "texel_sites_lr_init": 0.0,
            "texel_height_lr_init": 0.0,
            "texel_sv_axis_lr_init": 0.0,
            "texel_sv_rgb_lr_init": 0.1,
            "texel_sv_rgb_lr_final": 0.1,
        }
    )
    optimizer = torch.optim.Adam(student.optimizer_param_groups(train_cfg), lr=float(train_cfg["lr"]))
    initial_sv_rgb = student.raw_texel_sv_rgb.detach().clone()
    initial_xy = student.raw_xy.detach().clone()

    first_l1 = None
    l1 = None
    for _step in range(31):
        rendered, _alpha = student(frame_indices, rays=rays)
        l1 = (rendered - target).abs().mean()
        if first_l1 is None:
            first_l1 = float(l1.detach().cpu())
        loss = l1 + 0.1 * (rendered - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert first_l1 is not None and l1 is not None
    final_l1 = float(l1.detach().cpu())
    assert final_l1 < 0.001
    assert final_l1 < first_l1 * 0.25
    assert float((student.raw_texel_sv_rgb.detach() - initial_sv_rgb).abs().mean().cpu()) > 0.01
    torch.testing.assert_close(student.raw_xy.detach().cpu(), initial_xy.cpu())


def test_powerfoam_metal_knn_adjacency_has_fixed_degree_and_no_self_edges() -> None:
    try:
        from train_powerfoam_metal import build_csr_adjacency
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    points = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [4.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    radii = torch.full((4,), 0.5)

    rows, offsets = build_csr_adjacency(points, radii, neighbor_count=2, mode="knn")

    assert rows.dtype == torch.int32
    assert offsets.tolist() == [0, 2, 4, 6, 8]
    for cell in range(points.shape[0]):
        start = int(offsets[cell])
        end = int(offsets[cell + 1])
        assert cell not in rows[start:end].tolist()


def _csr_neighbors(rows: torch.Tensor, offsets: torch.Tensor, cell: int) -> list[int]:
    return rows[int(offsets[cell]) : int(offsets[cell + 1])].tolist()


def _fully_connected_csr(cell_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[int] = []
    offsets = [0]
    for cell in range(cell_count):
        rows.extend(other for other in range(cell_count) if other != cell)
        offsets.append(len(rows))
    return torch.tensor(rows, dtype=torch.int32), torch.tensor(offsets, dtype=torch.int32)


def _csr_edges(rows: torch.Tensor, offsets: torch.Tensor) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for cell in range(int(offsets.numel()) - 1):
        for neighbor in _csr_neighbors(rows, offsets, cell):
            if cell != int(neighbor):
                edges.add((min(cell, int(neighbor)), max(cell, int(neighbor))))
    return edges


def _single_center_ray_render(
    points: torch.Tensor,
    radii: torch.Tensor,
    densities: torch.Tensor,
    features: torch.Tensor,
    rows: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    origin = torch.zeros(3, dtype=torch.float32)
    direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    power = (points - origin).square().sum(dim=-1) - radii.square()
    sorted_ids = torch.argsort(power, stable=True).tolist()
    out = torch.zeros(features.shape[-1], dtype=torch.float32)
    transmittance = torch.tensor(1.0)
    eps = 1.0e-6
    for cell in sorted_ids:
        center = points[cell]
        radius = radii[cell]
        oc = origin - center
        b = 2.0 * torch.dot(oc, direction)
        c = torch.dot(oc, oc) - radius * radius
        disc = b * b - 4.0 * c
        if disc < 0.0:
            continue
        root = torch.sqrt(disc.clamp_min(0.0))
        t0 = (-b - root) * 0.5
        t1 = (-b + root) * 0.5
        if t1 <= 0.05:
            continue
        t0 = torch.maximum(t0, torch.tensor(0.05))
        inside = True
        for edge in range(int(offsets[cell]), int(offsets[cell + 1])):
            neighbor = int(rows[edge])
            if neighbor < 0 or neighbor >= points.shape[0] or neighbor == cell:
                continue
            neighbor_center = points[neighbor]
            nvec = neighbor_center - center
            rhs = (
                torch.dot(neighbor_center, neighbor_center)
                - torch.dot(center, center)
                + radius.square()
                - radii[neighbor].square()
            )
            limit = rhs - 2.0 * torch.dot(origin, nvec)
            denom = 2.0 * torch.dot(direction, nvec)
            if denom.abs() <= eps:
                if limit < -eps:
                    inside = False
                    break
                continue
            split = limit / denom
            if denom > 0.0:
                t1 = torch.minimum(t1, split)
            else:
                t0 = torch.maximum(t0, split)
            if t1 <= t0:
                inside = False
                break
        if not inside or t1 <= t0:
            continue
        alpha = (1.0 - torch.exp(-densities[cell].clamp_min(0.0) * (t1 - t0))).clamp(0.0, 0.99)
        out += transmittance * alpha * features[cell]
        transmittance = transmittance * (1.0 - alpha)
    return out


def test_powerfoam_metal_cech_aabb_is_dense_overlap_superset() -> None:
    try:
        from train_powerfoam_metal import build_csr_adjacency, csr_adjacency_stats
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    generator = torch.Generator().manual_seed(23)
    points = torch.randn(14, 3, generator=generator) * 0.45
    points[:, 2] += 2.0
    radii = torch.rand(14, generator=generator) * 0.28 + 0.18

    rows, offsets = build_csr_adjacency(points, radii, neighbor_count=1, mode="cech_aabb")
    stats = csr_adjacency_stats(points, radii, rows, offsets)

    assert stats["adjacency_missing_overlap_edges"] == 0.0
    assert rows.dtype == torch.int32
    for cell in range(points.shape[0]):
        assert cell not in _csr_neighbors(rows, offsets, cell)


def test_powerfoam_metal_cech_aabb_fixes_knn_missed_power_face() -> None:
    try:
        from train_powerfoam_metal import build_csr_adjacency
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    points = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.2, 0.0, 2.0],
            [0.0, 0.0, 2.8],
        ],
        dtype=torch.float32,
    )
    radii = torch.tensor([0.7, 0.05, 0.7], dtype=torch.float32)
    densities = torch.full((3,), 2.0, dtype=torch.float32)
    features = torch.eye(3, dtype=torch.float32)

    knn_rows, knn_offsets = build_csr_adjacency(points, radii, neighbor_count=1, mode="knn")
    cech_rows, cech_offsets = build_csr_adjacency(points, radii, neighbor_count=1, mode="cech_aabb")
    dense_rows, dense_offsets = _fully_connected_csr(points.shape[0])

    assert 2 not in _csr_neighbors(knn_rows, knn_offsets, 0)
    assert 2 in _csr_neighbors(cech_rows, cech_offsets, 0)

    knn = _single_center_ray_render(points, radii, densities, features, knn_rows, knn_offsets)
    cech = _single_center_ray_render(points, radii, densities, features, cech_rows, cech_offsets)
    dense = _single_center_ray_render(points, radii, densities, features, dense_rows, dense_offsets)

    assert torch.allclose(cech, dense, atol=1.0e-6)
    assert float((knn - dense).abs().max()) > 1.0e-3


def test_powerfoam_metal_regular_triangulation_matches_unweighted_delaunay_edges() -> None:
    if importlib.util.find_spec("scipy") is None:
        pytest.skip("regular_triangulation adjacency requires scipy")
    try:
        from scipy.spatial import Delaunay
        from train_powerfoam_metal import build_csr_adjacency
    except Exception as exc:  # pragma: no cover - depends on local optional scipy / Metal import path.
        pytest.skip(f"regular triangulation unavailable: {exc}")

    points = torch.tensor(
        [
            [-0.7, -0.4, 1.6],
            [0.7, -0.3, 1.8],
            [-0.4, 0.8, 2.0],
            [0.5, 0.7, 2.2],
            [0.0, 0.1, 2.8],
            [0.2, -0.9, 2.5],
            [-0.9, 0.2, 2.4],
            [0.9, 0.2, 2.7],
        ],
        dtype=torch.float32,
    )
    radii = torch.zeros(points.shape[0], dtype=torch.float32)

    rows, offsets = build_csr_adjacency(points, radii, neighbor_count=1, mode="regular_triangulation")
    actual = _csr_edges(rows.cpu(), offsets.cpu())

    expected: set[tuple[int, int]] = set()
    for simplex in Delaunay(points.numpy()).simplices:
        for i, a in enumerate(simplex):
            for b in simplex[i + 1 :]:
                expected.add((min(int(a), int(b)), max(int(a), int(b))))

    assert actual == expected
    assert all(cell not in _csr_neighbors(rows, offsets, cell) for cell in range(points.shape[0]))


def test_powerfoam_metal_resample_uses_ema_and_preserves_optimizer_state() -> None:
    try:
        from train_powerfoam_metal import MetalPowerFoamVideo, make_raster_config, RENDER_DEFAULTS, TRAIN_DEFAULTS
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    model = _make_test_metal_model(MetalPowerFoamVideo, make_raster_config, RENDER_DEFAULTS)
    optimizer = torch.optim.Adam(model.optimizer_param_groups(TRAIN_DEFAULTS), lr=1.0e-3)
    loss = model.raw_xy.square().sum() + model.raw_z.square().sum() + model.raw_radii.square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    old_radii = model.raw_radii.detach().clone()
    old_xy_state = optimizer.state[model.raw_xy]["exp_avg"].detach().clone()
    with torch.no_grad():
        model.contrib_ema[0] = torch.tensor([1.0e-9, 1.0, 1.0, 1.0])
        model.point_error_ema[0] = torch.tensor([0.0, 0.0, 10.0, 0.0])

    stats = model.resample_from_ema(optimizer, perturb_scale=0.0)

    expected = torch.tensor([1, 2, 3, 2])
    assert stats["resample_replaced"] == 1.0
    assert torch.allclose(model.raw_radii.detach()[0], old_radii[0, expected])
    assert torch.allclose(optimizer.state[model.raw_xy]["exp_avg"].detach()[0], old_xy_state[0, expected])
    assert model.contrib_ema.shape == (1, 4)
    assert model.point_error_ema.shape == (1, 4)

    with torch.no_grad():
        model.contrib_ema[0] = torch.tensor([1.0e-9, 1.0, 1.0, 1.0])
        model.point_error_ema[0] = torch.tensor([0.0, 2.0, 10.0, 4.0])
    grow_stats = model.resample_from_ema(optimizer, target_cells=6, perturb_scale=0.0)
    assert grow_stats["resample_cell_count"] == 6.0
    assert model.raw_radii.shape == (1, 6)
    assert optimizer.state[model.raw_xy]["exp_avg"].shape == model.raw_xy.shape
    assert model.contrib_ema.shape == (1, 6)
    assert model.parameter_drift_metrics()["state_cell_count"] == 6.0

    with torch.no_grad():
        model.contrib_ema[0] = torch.tensor([0.0, 0.5, 0.1, 0.8, 0.2, 0.7])
        model.point_error_ema[0] = torch.arange(6, dtype=torch.float32)
    prune_stats = model.resample_from_ema(optimizer, target_cells=3, perturb_scale=0.0)
    assert prune_stats["resample_cell_count"] == 3.0
    assert model.raw_radii.shape == (1, 3)
    assert optimizer.state[model.raw_xy]["exp_avg"].shape == model.raw_xy.shape
    assert model.contrib_ema.shape == (1, 3)
    assert model.parameter_drift_metrics()["state_cell_count"] == 3.0


def test_powerfoam_metal_resample_prunes_invalid_cells() -> None:
    try:
        from train_powerfoam_metal import MetalPowerFoamVideo, make_raster_config, RENDER_DEFAULTS, TRAIN_DEFAULTS
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    model = _make_test_metal_model(
        MetalPowerFoamVideo,
        make_raster_config,
        RENDER_DEFAULTS,
        z_max=3.0,
        seed=9,
    )
    optimizer = torch.optim.Adam(model.optimizer_param_groups(TRAIN_DEFAULTS), lr=1.0e-3)
    with torch.no_grad():
        model.raw_features[0, 0, 0] = float("nan")
        model.contrib_ema[0] = torch.tensor([100.0, 0.5, 0.4, 0.3])
        model.point_error_ema[0] = torch.tensor([100.0, 1.0, 2.0, 3.0])

    stats = model.resample_from_ema(optimizer, target_cells=3, perturb_scale=0.0)

    assert stats["resample_invalid_pruned"] == 1.0
    assert model.raw_features.shape == (1, 3, 3)
    assert torch.isfinite(model.raw_features).all()
    assert model.parameter_drift_metrics()["state_cell_count"] == 3.0


def test_camera_facing_quaternion_points_normals_toward_camera() -> None:
    normals, _, _ = quaternion_frames(camera_facing_quaternion(frame_count=1, cell_count=2))
    assert torch.allclose(normals[..., 2], -torch.ones_like(normals[..., 2]), atol=1.0e-6)


def test_powerfoam_regularizer_weights_use_official_exp_decay_shape() -> None:
    start = scheduled_loss_weights(LOSS_DEFAULTS, step=0, total_steps=100)
    end = scheduled_loss_weights(LOSS_DEFAULTS, step=100, total_steps=100)
    assert abs(start["normal_weight"] - 0.1) < 1.0e-8
    assert abs(end["normal_weight"] - 0.01) < 1.0e-8
    assert abs(end["contribution_weight"] - 0.0001) < 1.0e-8
    assert abs(end["interpenetration_weight"] - 1.0e-7) < 1.0e-10


def test_powerfoam_metal_lr_schedule_uses_official_cosine_shape_and_warmups() -> None:
    try:
        from train_powerfoam_metal import (
            TRAIN_DEFAULTS,
            cosine_scheduled_lr,
            powerfoam_group_lr_metadata,
            update_powerfoam_learning_rates,
        )
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    assert abs(cosine_scheduled_lr(1.0e-3, 5.0e-5, 50, 100) - 5.25e-4) < 1.0e-10

    train_cfg = dict(TRAIN_DEFAULTS)
    train_cfg.update(
        {
            "lr_schedule": "cosine",
            "points_lr_init": 1.0e-3,
            "points_lr_final": 5.0e-5,
            "density_lr_init": 1.0,
            "density_lr_final": 1.0,
        }
    )
    point_meta = powerfoam_group_lr_metadata(train_cfg, "points")
    density_meta = powerfoam_group_lr_metadata(train_cfg, "density")
    assert point_meta["lr"] == 1.0e-3
    assert point_meta["final_lr"] == 5.0e-5
    assert density_meta["warmup_steps"] == 1000
    train_cfg["lr_warmup_steps"] = {"density": 20}
    density_meta_override = powerfoam_group_lr_metadata(train_cfg, "density")
    assert density_meta_override["warmup_steps"] == 20
    train_cfg["lr_warmup_steps"] = {}

    point = torch.nn.Parameter(torch.ones(()))
    density = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam(
        [
            {"params": [point], "name": "points", **point_meta},
            {"params": [density], "name": "density", **density_meta},
        ]
    )
    lrs = update_powerfoam_learning_rates(optimizer, train_cfg, step=500, total_steps=2000)
    assert abs(lrs["density"] - 0.5) < 1.0e-8
    lrs = update_powerfoam_learning_rates(optimizer, train_cfg, step=100, total_steps=100)
    assert abs(lrs["points"] - 5.0e-5) < 1.0e-10


def test_powerfoam_metal_contribution_loss_uses_differentiable_alpha_sum() -> None:
    try:
        from train_powerfoam_metal import LOSS_DEFAULTS as METAL_LOSS_DEFAULTS, powerfoam_contribution_loss, scheduled_loss_weights
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    alpha = torch.tensor([[[0.0, 0.5], [0.75, 1.0]]], dtype=torch.float32, requires_grad=True)
    loss = powerfoam_contribution_loss(alpha)
    assert abs(float(loss.detach()) - 0.5625) < 1.0e-8
    loss.backward()
    assert torch.allclose(alpha.grad, torch.full_like(alpha, 0.25))

    loss_cfg = dict(METAL_LOSS_DEFAULTS)
    loss_cfg.update(
        {
            "contribution_weight": 0.1,
            "contribution_weight_final_multiplier": 0.001,
            "normal_map_weight": 0.1,
            "normal_map_weight_final_multiplier": 0.5,
        }
    )
    weights = scheduled_loss_weights(loss_cfg, step=100, total_steps=100)
    assert abs(weights["contribution_weight"] - 0.0001) < 1.0e-8
    assert abs(weights["normal_map_weight"] - 0.05) < 1.0e-8


def test_powerfoam_normals_from_ray_depth_orients_against_rays() -> None:
    try:
        from train_powerfoam_metal import make_pinhole_rays, normals_from_ray_depth
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    rays = make_pinhole_rays(7, 7, 55.0, torch.device("cpu"))
    depth = 2.0 / rays[..., 5].clamp_min(1.0e-6)
    normals, mask = normals_from_ray_depth(depth, rays)
    assert mask[:, 1:-1, 1:-1].all()
    assert not mask[:, 0].any()
    assert torch.allclose(normals[0, 3, 3], torch.tensor([0.0, 0.0, -1.0]), atol=2.0e-4)
    assert (normals[mask] * rays.expand_as(rays)[..., 3:][mask]).sum(dim=-1).max() <= 1.0e-5


def test_powerfoam_normal_map_loss_masks_invalid_pixels() -> None:
    try:
        from train_powerfoam_metal import powerfoam_normal_map_loss
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    rendered = torch.zeros(1, 2, 2, 3, requires_grad=True)
    target = torch.zeros_like(rendered)
    rendered.data[0, 0, 0, 0] = 10.0
    rendered.data[0, 1, 1, 1] = 2.0
    mask = torch.tensor([[[False, False], [False, True]]])
    loss = powerfoam_normal_map_loss(rendered, target, mask)
    assert abs(float(loss.detach()) - 4.0) < 1.0e-8
    loss.backward()
    assert rendered.grad is not None
    assert float(rendered.grad[0, 0, 0].abs().sum()) == 0.0
    assert float(rendered.grad[0, 1, 1].abs().sum()) > 0.0


def test_powerfoam_metal_normal_distance_loss_backprops_through_tiled_primitive() -> None:
    try:
        from train_powerfoam_metal import (
            MetalPowerFoamVideo,
            RENDER_DEFAULTS,
            make_raster_config,
            powerfoam_normal_distance_loss,
        )
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")
    if not torch.backends.mps.is_available():
        pytest.skip("PowerFoam Metal normal-distance loss requires MPS.")

    device = torch.device("mps")
    model = _make_test_metal_model(
        MetalPowerFoamVideo,
        make_raster_config,
        RENDER_DEFAULTS,
        cell_count=1,
        render_size=8,
        neighbor_count=0,
        radius_init=0.6,
        density_init=8.0,
        feature_mode="quaternion_height_sv_texel_surface",
        raster_overrides={"use_tiled": True},
    ).to(device)
    with torch.no_grad():
        model.raw_xy.zero_()
        model.raw_z.zero_()
        model.raw_radii.fill_(0.6)
        model.raw_densities.fill_(8.0)
        model.raw_quaternions[0, 0] = torch.tensor(
            [math.cos(math.pi / 8.0), 0.0, math.sin(math.pi / 8.0), 0.0],
            device=device,
        )

    rendered, alpha, normal_distance = model(torch.tensor([0], device=device), return_normal_distance=True)
    assert rendered.shape == (1, 3, 8, 8)
    assert alpha.shape == (1, 8, 8)
    assert normal_distance.shape == (1, 8, 8)
    loss = powerfoam_normal_distance_loss(normal_distance)
    assert float(loss.detach().cpu()) > 0.0
    loss.backward()
    assert model.raw_quaternions.grad is not None and torch.isfinite(model.raw_quaternions.grad).all()
    assert model.raw_densities.grad is not None and torch.isfinite(model.raw_densities.grad).all()
    assert float(model.raw_quaternions.grad.abs().sum().detach().cpu()) > 0.0
    assert float(model.raw_densities.grad.abs().sum().detach().cpu()) > 0.0


def test_powerfoam_metal_raytrace_rendered_normal_backprops() -> None:
    try:
        from train_powerfoam_metal import MetalPowerFoamVideo, RENDER_DEFAULTS, make_raster_config
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")
    if not torch.backends.mps.is_available():
        pytest.skip("PowerFoam Metal rendered-normal raytrace loss requires MPS.")

    device = torch.device("mps")
    model = _make_test_metal_model(
        MetalPowerFoamVideo,
        make_raster_config,
        RENDER_DEFAULTS,
        cell_count=1,
        render_size=8,
        neighbor_count=0,
        radius_init=0.7,
        density_init=8.0,
        feature_mode="quaternion_height_sv_texel_surface",
        use_raytrace=True,
    ).to(device)
    with torch.no_grad():
        model.raw_xy.zero_()
        model.raw_z.zero_()
        model.raw_radii.fill_(0.7)
        model.raw_densities.fill_(8.0)
        model.raw_quaternions[0, 0] = torch.tensor(
            [math.cos(math.pi / 8.0), 0.0, math.sin(math.pi / 8.0), 0.0],
            device=device,
        )

    rendered, alpha, normal_distance, rendered_normal = model(
        torch.tensor([0], device=device),
        return_normal_distance=True,
        return_rendered_normal=True,
    )
    assert rendered.shape == (1, 3, 8, 8)
    assert alpha.shape == (1, 8, 8)
    assert normal_distance.shape == (1, 8, 8)
    assert rendered_normal.shape == (1, 8, 8, 3)
    loss = rendered_normal.square().mean() + 0.01 * normal_distance.mean()
    assert float(loss.detach().cpu()) > 0.0
    loss.backward()
    assert model.raw_quaternions.grad is not None and torch.isfinite(model.raw_quaternions.grad).all()
    assert model.raw_densities.grad is not None and torch.isfinite(model.raw_densities.grad).all()
    assert float(model.raw_quaternions.grad.abs().sum().detach().cpu()) > 0.0
    assert float(model.raw_densities.grad.abs().sum().detach().cpu()) > 0.0


def test_powerfoam_metal_raytrace_height_sv_backward_supports_9_texel_sites() -> None:
    try:
        from train_powerfoam_metal import MetalPowerFoamVideo, RENDER_DEFAULTS, make_raster_config
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")
    if not torch.backends.mps.is_available():
        pytest.skip("PowerFoam Metal high-capacity raytrace backward requires MPS.")

    device = torch.device("mps")
    model = _make_test_metal_model(
        MetalPowerFoamVideo,
        make_raster_config,
        RENDER_DEFAULTS,
        cell_count=1,
        render_size=8,
        neighbor_count=0,
        radius_init=0.7,
        density_init=8.0,
        feature_mode="quaternion_height_sv_texel_surface",
        num_texel_sites=9,
        sv_dof=3,
        use_raytrace=True,
    ).to(device)
    with torch.no_grad():
        model.raw_xy.zero_()
        model.raw_z.zero_()
        model.raw_radii.fill_(0.7)
        model.raw_densities.fill_(8.0)

    rendered, alpha, normal_distance, rendered_normal = model(
        torch.tensor([0], device=device),
        return_normal_distance=True,
        return_rendered_normal=True,
    )
    assert rendered.shape == (1, 3, 8, 8)
    assert alpha.shape == (1, 8, 8)
    assert normal_distance.shape == (1, 8, 8)
    assert rendered_normal.shape == (1, 8, 8, 3)
    loss = rendered.square().mean() + 0.01 * alpha.mean() + 0.01 * normal_distance.mean() + 0.01 * rendered_normal.square().mean()
    loss.backward()
    assert model.raw_texel_sv_rgb.grad is not None and torch.isfinite(model.raw_texel_sv_rgb.grad).all()
    assert model.raw_texel_sv_axis.grad is not None and torch.isfinite(model.raw_texel_sv_axis.grad).all()
    assert model.raw_texel_sites.grad is not None and torch.isfinite(model.raw_texel_sites.grad).all()
    assert model.raw_quaternions.grad is not None and torch.isfinite(model.raw_quaternions.grad).all()
    assert float(model.raw_texel_sv_rgb.grad.abs().sum().detach().cpu()) > 0.0


def test_powerfoam_metal_normal_map_loss_uses_aux_median_depth_without_metric3d() -> None:
    try:
        from train_powerfoam_metal import (
            MetalPowerFoamVideo,
            RENDER_DEFAULTS,
            make_raster_config,
            normals_from_ray_depth,
            powerfoam_normal_map_loss,
        )
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")
    if not torch.backends.mps.is_available():
        pytest.skip("PowerFoam Metal normal-map loss requires MPS.")

    device = torch.device("mps")
    model = _make_test_metal_model(
        MetalPowerFoamVideo,
        make_raster_config,
        RENDER_DEFAULTS,
        cell_count=1,
        render_size=8,
        neighbor_count=0,
        radius_init=0.7,
        density_init=8.0,
        feature_mode="quaternion_height_sv_texel_surface",
        use_raytrace=True,
    ).to(device)
    with torch.no_grad():
        model.raw_xy.zero_()
        model.raw_z.zero_()
        model.raw_radii.fill_(0.7)
        model.raw_densities.fill_(8.0)
        model.raw_quaternions[0, 0] = torch.tensor(
            [math.cos(math.pi / 8.0), 0.0, math.sin(math.pi / 8.0), 0.0],
            device=device,
        )

    frame_indices = torch.tensor([0], device=device)
    rendered, alpha, rendered_normal = model(frame_indices, return_rendered_normal=True)
    aux = model.height_sv_aux_batch(frame_indices)
    assert aux is not None
    target_normal, valid_mask = normals_from_ray_depth(aux.median_depth, model.rays.to(device=device))
    valid_mask = valid_mask & (alpha.detach() >= 0.05)
    assert bool(valid_mask.any().detach().cpu())
    loss = powerfoam_normal_map_loss(rendered_normal, target_normal.detach(), valid_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.raw_quaternions.grad is not None and torch.isfinite(model.raw_quaternions.grad).all()
    assert model.raw_densities.grad is not None and torch.isfinite(model.raw_densities.grad).all()
    assert float(model.raw_quaternions.grad.abs().sum().detach().cpu()) > 0.0
    assert float(model.raw_densities.grad.abs().sum().detach().cpu()) > 0.0


def test_powerfoam_metal_interpenetration_loss_is_differentiable() -> None:
    try:
        from train_powerfoam_metal import (
            LOSS_DEFAULTS as METAL_LOSS_DEFAULTS,
            MetalPowerFoamVideo,
            RENDER_DEFAULTS,
            make_raster_config,
            scheduled_loss_weights,
        )
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    model = _make_test_metal_model(
        MetalPowerFoamVideo,
        make_raster_config,
        RENDER_DEFAULTS,
        cell_count=2,
        neighbor_count=1,
    )
    with torch.no_grad():
        xy = torch.tensor([[[-0.05, 0.0], [0.05, 0.0]]], dtype=torch.float32)
        model.raw_xy.copy_(torch.atanh((xy / model.xy_extent).clamp(-0.9999, 0.9999)))
        model.raw_z.zero_()
        decoded_radius = torch.full((1, 2), 0.3 - model.radius_min)
        model.raw_radii.copy_(inverse_softplus(decoded_radius, beta=POWERFOAM_SOFTPLUS_BETA))

    loss = model.interpenetration_loss(torch.tensor([0]))
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert model.raw_xy.grad is not None and torch.isfinite(model.raw_xy.grad).all()
    assert model.raw_radii.grad is not None and torch.isfinite(model.raw_radii.grad).all()
    assert float(model.raw_xy.grad.abs().sum()) > 0.0
    assert float(model.raw_radii.grad.abs().sum()) > 0.0

    loss_cfg = dict(METAL_LOSS_DEFAULTS)
    loss_cfg.update({"interpenetration_weight": 1.0e-4, "interpenetration_weight_final_multiplier": 0.001})
    weights = scheduled_loss_weights(loss_cfg, step=100, total_steps=100)
    assert abs(weights["interpenetration_weight"] - 1.0e-7) < 1.0e-10


def test_powerfoam_metal_resample_schedule_matches_official_geometric_growth() -> None:
    try:
        from train_powerfoam_metal import MODEL_DEFAULTS, scheduled_resample_target_cells, should_resample_powerfoam_step
    except Exception as exc:  # pragma: no cover - depends on local Metal extension build.
        pytest.skip(f"powerfoam_metal unavailable: {exc}")

    model_cfg = dict(MODEL_DEFAULTS)
    model_cfg.update(
        {
            "resample_target_cells": None,
            "resample_final_cells": 16,
            "resample_from_step": 2,
            "resample_until_step": 6,
        }
    )
    assert scheduled_resample_target_cells(model_cfg, initial_cells=4, current_cells=4, step=1, total_steps=10) == 4
    assert scheduled_resample_target_cells(model_cfg, initial_cells=4, current_cells=4, step=2, total_steps=10) == 4
    assert scheduled_resample_target_cells(model_cfg, initial_cells=4, current_cells=6, step=3, total_steps=10) == 6
    assert scheduled_resample_target_cells(model_cfg, initial_cells=4, current_cells=10, step=4, total_steps=10) == 10
    assert scheduled_resample_target_cells(model_cfg, initial_cells=4, current_cells=10, step=5, total_steps=10) == 15
    assert scheduled_resample_target_cells(model_cfg, initial_cells=4, current_cells=16, step=6, total_steps=10) == 16

    model_cfg["resample_target_cells"] = 7
    assert scheduled_resample_target_cells(model_cfg, initial_cells=4, current_cells=4, step=3, total_steps=10) == 7

    train_cfg = {"model": {"resample_every": 2}, "train": {"steps": 5}, "logging": {"image_log_every": 999}}
    assert not should_resample_powerfoam_step(train_cfg, 1)
    assert should_resample_powerfoam_step(train_cfg, 2)
    assert should_resample_powerfoam_step(train_cfg, 4)
    assert not should_resample_powerfoam_step(train_cfg, 5)


def test_powerfoam_video_init_samples_frame_colors() -> None:
    frames = torch.zeros(2, 3, 8, 8)
    frames[:, 0] = torch.linspace(0.0, 1.0, 8).view(1, 1, 8)
    frames[:, 1] = torch.linspace(0.0, 1.0, 8).view(1, 8, 1)
    points, colors = initialize_powerfoam_from_video(
        frames,
        cell_count=4,
        xy_extent=1.25,
        z_min=1.0,
        z_max=3.0,
        fov_degrees=55.0,
        image_init_depth=2.0,
    )

    assert points.shape == (2, 4, 3)
    assert colors.shape == (2, 4, 3)
    assert torch.isfinite(points).all()
    assert torch.isfinite(colors).all()
    assert torch.all((colors >= 0.0) & (colors <= 1.0))
    assert colors[..., 0].amin() < colors[..., 0].amax()
    assert colors[..., 1].amin() < colors[..., 1].amax()


def test_powerfoam_image_init_jitter_breaks_exact_lattice_deterministically() -> None:
    frames = torch.rand(1, 3, 8, 8)
    base_points, _ = initialize_powerfoam_from_video(
        frames,
        cell_count=16,
        xy_extent=1.25,
        z_min=1.0,
        z_max=3.0,
        fov_degrees=55.0,
        image_init_depth=2.0,
    )
    gen_a = torch.Generator().manual_seed(11)
    gen_b = torch.Generator().manual_seed(11)
    jittered_a, _ = initialize_powerfoam_from_video(
        frames,
        cell_count=16,
        xy_extent=1.25,
        z_min=1.0,
        z_max=3.0,
        fov_degrees=55.0,
        image_init_depth=2.0,
        image_init_jitter=0.5,
        generator=gen_a,
    )
    jittered_b, _ = initialize_powerfoam_from_video(
        frames,
        cell_count=16,
        xy_extent=1.25,
        z_min=1.0,
        z_max=3.0,
        fov_degrees=55.0,
        image_init_depth=2.0,
        image_init_jitter=0.5,
        generator=gen_b,
    )

    assert torch.allclose(jittered_a, jittered_b)
    assert not torch.allclose(base_points[..., :2], jittered_a[..., :2])
