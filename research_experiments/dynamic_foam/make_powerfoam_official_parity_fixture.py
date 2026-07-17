from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.nn import functional as F

from powerfoam_direct import PowerFoamRenderOptions, render_powerfoam_torch

try:
    from .report_artifacts import load_report_json, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import load_report_json, write_report_json


UPSTREAM_POWERFOAM_COMMIT = "96392252ebd0059fe6ca98881b62e12295d9242f"
OFFICIAL_RASTER_TEXEL_TEMPERATURE = 10.0

FLOAT_SCENE_KEYS = (
    "points",
    "radii",
    "densities",
    "quaternions",
    "normals",
    "local_texel_sites",
    "texel_sites",
    "texel_height",
    "texel_sv_axis",
    "texel_sv_rgb",
)
SCENE_TENSOR_KEYS = (*FLOAT_SCENE_KEYS, "adjacency")
REQUIRES_GRAD_KEYS = (
    "points",
    "radii",
    "densities",
    "normals",
    "texel_sites",
    "texel_height",
    "texel_sv_axis",
    "texel_sv_rgb",
)
GRAD_OUTPUT_NAMES = {
    "densities": "grad_density",
}


@dataclass(frozen=True)
class OfficialCameraSpec:
    eye: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    width: int
    height: int


def tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "values": tensor.detach().cpu().tolist(),
    }


def tensor_from_payload(payload: dict[str, Any], *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.tensor(payload["values"])
    if dtype is not None:
        return tensor.to(dtype=dtype)
    if payload.get("dtype") == "float32":
        return tensor.to(dtype=torch.float32)
    if payload.get("dtype") in {"int32", "int64"}:
        return tensor.to(dtype=torch.long)
    return tensor


def load_local_scene(fixture_path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor], PowerFoamRenderOptions]:
    fixture = load_report_json(fixture_path)
    inputs = fixture["inputs"]
    scene = {key: tensor_from_payload(inputs[key], dtype=torch.float32) for key in FLOAT_SCENE_KEYS}
    scene["adjacency"] = tensor_from_payload(inputs["adjacency"]).to(dtype=torch.long)
    options = PowerFoamRenderOptions(**fixture["render_options"])
    return fixture, scene, options


def official_camera_payload() -> OfficialCameraSpec:
    # Upstream PowerFoam's TorchCamera uses forward = normalize(cross(up, right)).
    # Negative X right vector makes the camera look along +Z.
    return OfficialCameraSpec(
        eye=(0.16, -0.08, -1.0),
        right=(-0.25, 0.0, 0.0),
        up=(0.0, 0.25, 0.0),
        width=3,
        height=3,
    )


def rays_from_official_camera(camera: OfficialCameraSpec, *, device: torch.device) -> torch.Tensor:
    eye = torch.tensor(camera.eye, dtype=torch.float32, device=device)
    right = torch.tensor(camera.right, dtype=torch.float32, device=device)
    up = torch.tensor(camera.up, dtype=torch.float32, device=device)
    forward = F.normalize(torch.cross(up, right, dim=0), dim=0)
    rows = []
    for i in range(camera.height):
        cols = []
        for j in range(camera.width):
            x = 2.0 * float(j) / float(camera.width - 1) - 1.0
            y = 1.0 - 2.0 * float(i) / float(camera.height - 1)
            direction = F.normalize(x * right + y * up + forward, dim=0)
            cols.append(torch.cat([eye, direction], dim=0))
        rows.append(torch.stack(cols, dim=0))
    return torch.stack(rows, dim=0).unsqueeze(0)


def padded_adjacency_to_csr(adjacency: torch.Tensor, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    padded = adjacency[0].to(dtype=torch.long)
    rows: list[int] = []
    offsets = [0]
    for cell in range(int(padded.shape[0])):
        for neighbor in padded[cell].tolist():
            if int(neighbor) >= 0:
                rows.append(int(neighbor))
        offsets.append(len(rows))
    return (
        torch.tensor(rows, dtype=torch.int32, device=device),
        torch.tensor(offsets, dtype=torch.int32, device=device),
    )


def grad_name(scene_key: str) -> str:
    return GRAD_OUTPUT_NAMES.get(scene_key, f"grad_{scene_key}")


def clone_grad_params(
    scene: dict[str, torch.Tensor],
    *,
    device: torch.device | None = None,
    batch_index: int | None = None,
) -> dict[str, torch.Tensor]:
    params = {}
    for key in REQUIRES_GRAD_KEYS:
        value = scene[key]
        if batch_index is not None:
            value = value[batch_index]
        if device is not None:
            value = value.to(device)
        params[key] = value.detach().clone().requires_grad_(True)
    return params


def grad_outputs(params: dict[str, torch.Tensor], *, batch_index: int | None = None) -> dict[str, torch.Tensor]:
    outputs = {}
    for key, param in params.items():
        if param.grad is None:
            raise RuntimeError(f"{key} did not receive a gradient.")
        grad = param.grad
        outputs[grad_name(key)] = grad[batch_index] if batch_index is not None else grad
    return outputs


def render_options_payload(options: PowerFoamRenderOptions) -> dict[str, Any]:
    return {
        "near_plane": options.near_plane,
        "alpha_threshold": options.alpha_threshold,
        "transmittance_threshold": options.transmittance_threshold,
        "max_alpha": options.max_alpha,
        "eps": options.eps,
        "texel_temperature": options.texel_temperature,
        "background": list(options.background),
    }


def local_outputs(scene: dict[str, torch.Tensor], options: PowerFoamRenderOptions, rays: torch.Tensor) -> dict[str, torch.Tensor]:
    params = clone_grad_params(scene)
    result = render_powerfoam_torch(
        params["points"],
        params["radii"],
        params["densities"],
        params["normals"],
        params["texel_sites"],
        None,
        params["texel_height"],
        scene["adjacency"],
        rays,
        options,
        texel_sv_axis=params["texel_sv_axis"],
        texel_sv_rgb=params["texel_sv_rgb"],
    )
    loss = result.rendered.square().mean() + result.alpha.square().mean() + result.normal_distance.square().mean()
    loss.backward()
    return {
        "rendered": result.rendered,
        "alpha": result.alpha,
        "normal_distance": result.normal_distance,
        "normal": result.normal,
        "contrib": result.contrib,
        "visible_mask": result.visible_mask.to(dtype=torch.int32),
        "loss": loss.detach().reshape(()),
        **grad_outputs(params, batch_index=0),
    }


def _official_attention_tensors(texel_sv_axis: torch.Tensor, texel_sv_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Local fixture layout: [N,S,D,3]. Upstream SphericalVoronoi expects
    # structure-of-arrays [D,N*S,3] plus [D,N*S] temperatures.
    axis = texel_sv_axis
    temp = axis.norm(dim=-1)
    axis = axis / temp.clamp_min(1.0e-8)[..., None]
    att_sites = axis.permute(2, 0, 1, 3).reshape(axis.shape[2], -1, 3).contiguous()
    att_values = texel_sv_rgb.permute(2, 0, 1, 3).reshape(axis.shape[2], -1, 3).contiguous()
    att_temps = temp.permute(2, 0, 1).reshape(axis.shape[2], -1).contiguous()
    return att_sites, att_values, att_temps


def official_outputs(
    *,
    upstream_root: Path,
    scene: dict[str, torch.Tensor],
    camera_spec: OfficialCameraSpec,
    options: PowerFoamRenderOptions,
) -> dict[str, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("Official PowerFoam fixture generation requires CUDA.")
    try:
        import warp
    except Exception as exc:  # pragma: no cover - only exercised on CUDA host.
        raise RuntimeError("Official PowerFoam fixture generation requires warp-lang.") from exc
    warp.init()

    upstream_root = upstream_root.resolve()
    sys.path.insert(0, str(upstream_root))
    camera_mod = importlib.import_module("powerfoam.camera")
    raster_mod = importlib.import_module("powerfoam.rasterize")
    color_mod = importlib.import_module("powerfoam.color_fn")

    device = torch.device("cuda")
    args = SimpleNamespace(
        is_pinhole=True,
        render_objective="rgb",
        num_texel_sites=int(scene["texel_sites"].shape[2]),
        sv_dof=int(scene["texel_sv_axis"].shape[3]),
        disable_coop_prim_load=False,
        disable_coop_adj_load=False,
    )
    camera = camera_mod.TorchCamera(
        eye=torch.tensor(camera_spec.eye, dtype=torch.float32, device=device),
        right=torch.tensor(camera_spec.right, dtype=torch.float32, device=device),
        up=torch.tensor(camera_spec.up, dtype=torch.float32, device=device),
        width=int(camera_spec.width),
        height=int(camera_spec.height),
    )
    rasterizer = raster_mod.Rasterizer(args, device, attr_dtype="float")
    sv = color_mod.SphericalVoronoi(args, device, attr_dtype="float")
    sv.fov_cos_cutoff = color_mod.SphericalVoronoi.compute_fov_cos_cutoff(camera)

    params = clone_grad_params(scene, device=device, batch_index=0)
    adjacency, adjacency_offsets = padded_adjacency_to_csr(scene["adjacency"], device=device)
    att_sites, att_values, att_temps = _official_attention_tensors(params["texel_sv_axis"], params["texel_sv_rgb"])
    texel_rgb = sv.forward(params["texel_sites"].reshape(-1, 3).detach(), camera, att_sites, att_values, att_temps)
    texel_rgb = texel_rgb.view(params["points"].shape[0], args.num_texel_sites, 3)

    color, alpha, normal_distance, normal, quantiles, err, contrib, point_err, visible = rasterizer.forward(
        camera,
        None,
        params["points"],
        params["radii"],
        params["densities"],
        params["normals"],
        params["texel_sites"],
        texel_rgb,
        params["texel_height"],
        adjacency,
        adjacency_offsets,
        None,
        False,
    )
    loss = color.square().mean() + alpha.square().mean() + normal_distance.square().mean()
    loss.backward()
    return {
        "rendered": color.permute(2, 0, 1).unsqueeze(0).detach().cpu(),
        "alpha": alpha.unsqueeze(0).detach().cpu(),
        "normal_distance": normal_distance.unsqueeze(0).detach().cpu(),
        "normal": normal.permute(2, 0, 1).unsqueeze(0).detach().cpu(),
        "contrib": contrib.unsqueeze(0).detach().cpu(),
        "visible_mask": visible.to(dtype=torch.int32).unsqueeze(0).detach().cpu(),
        "loss": loss.detach().cpu().reshape(()),
        **{key: value.detach().cpu() for key, value in grad_outputs(params).items()},
    }


def make_payload(
    *,
    backend: str,
    upstream_root: Path,
    fixture_path: Path,
) -> dict[str, Any]:
    source_fixture, scene, options = load_local_scene(fixture_path)
    camera = official_camera_payload()
    rays = rays_from_official_camera(camera, device=torch.device("cpu"))
    if backend == "local":
        expected = local_outputs(scene, options, rays)
    elif backend == "official":
        # Upstream PowerFoam's raster texture path hard-codes temp=10.0 at this
        # pin, so the generated fixture must expose that effective option for
        # local Direct/Metal parity checks.
        options = replace(options, texel_temperature=OFFICIAL_RASTER_TEXEL_TEMPERATURE)
        expected = official_outputs(
            upstream_root=upstream_root,
            scene=scene,
            camera_spec=camera,
            options=options,
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}.")

    return {
        "metadata": {
            "name": f"powerfoam_tiny_height_sv_official_camera_{backend}_v1",
            "backend": backend,
            "upstream_powerfoam_commit": UPSTREAM_POWERFOAM_COMMIT,
            "source_fixture": str(fixture_path),
                "notes": (
                    "Official backend must be generated on a CUDA/Warp host from the "
                    "pinned upstream checkout. Local backend is a Mac-runnable dry run "
                    "for the same official-compatible pinhole camera fixture. The "
                    "official backend records the upstream raster texture temperature "
                    "that is hard-coded in the pinned CUDA/Warp path."
                ),
                "official_raster_texel_temperature": OFFICIAL_RASTER_TEXEL_TEMPERATURE
                if backend == "official"
                else None,
            },
        "official_camera": {
            "eye": list(camera.eye),
            "right": list(camera.right),
            "up": list(camera.up),
            "width": camera.width,
            "height": camera.height,
        },
        "render_options": render_options_payload(options),
        "inputs": {**{key: tensor_payload(scene[key]) for key in SCENE_TENSOR_KEYS}, "rays": tensor_payload(rays)},
        "expected": {key: tensor_payload(value) for key, value in expected.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "official"], default="local")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_origin_parity_v1.json"),
    )
    parser.add_argument("--upstream-root", type=Path, default=Path("/tmp/powerfoam_official"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    payload = make_payload(backend=args.backend, upstream_root=args.upstream_root, fixture_path=args.fixture)
    output = args.output
    if output is None:
        output = Path(f"research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_{args.backend}_v1.json")
    write_report_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
