from __future__ import annotations

import argparse
import itertools
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parents[1]
TRAIN_DIR = PROJECT_ROOT / "src" / "train"
VENDORED_TAICHI_SPLATTING_DIR = PROJECT_ROOT / "third_party" / "taichi-splatting"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
if VENDORED_TAICHI_SPLATTING_DIR.exists() and str(VENDORED_TAICHI_SPLATTING_DIR) not in sys.path:
    sys.path.insert(0, str(VENDORED_TAICHI_SPLATTING_DIR))

from config_utils import load_config_file
from raw_metal_mlx_bridge import RawMetalUnavailable, import_raw_metal, run_packed_accuracy

DEFAULT_CONFIG: dict[str, Any] = {
    "device": "auto",
    "dtype": "float32",
    "baseline_device": "cpu",
    "baseline_dtype": "float64",
    "renderers": ["taichi", "raw_metal"],
    "resolutions": [[16, 16], [32, 32]],
    "splat_counts": [4, 8, 16],
    "sets_per_case": 3,
    "seed": 20260420,
    "random_splats_2d": {
        "xy_margin_fraction": 0.12,
        "sigma_pixel_range": [1.25, 4.5],
        "opacity_range": [0.04, 0.35],
        "feature_range": [0.05, 0.95],
        "background": [0.08, 0.13, 0.21],
    },
    "taichi": {
        "variant": "metal_reference",
        "sort_backend": "auto",
        "backward_variant": "pixel_reference",
        "metal_block_dim": 0,
        "use_depth16": False,
        "tile_size": 16,
        "alpha_threshold": 1.0 / 255.0,
        "clamp_max_alpha": 0.99,
        "saturate_threshold": 0.9999,
        "metal_compatible": True,
    },
    "raw_metal": {
        "tile_size": 16,
        "chunk_size": 32,
        "alpha_threshold": 1.0 / 255.0,
        "transmittance_threshold": 1.0e-4,
    },
    "tolerances": {
        "image_max_abs": 5.0e-4,
        "image_mean_abs": 5.0e-5,
        "loss_abs": 5.0e-5,
        "packed_grad_max_abs": 2.0e-3,
        "packed_grad_mean_abs": 1.0e-4,
        "feature_grad_max_abs": 2.0e-3,
        "feature_grad_mean_abs": 1.0e-4,
    },
    "save_images": {
        "enabled": True,
        "directory": "benchmark_outputs/splat_accuracy_images",
        "set_index": 0,
        "largest_resolution_only": True,
        "largest_splat_count_only": True,
    },
}


@dataclass(frozen=True)
class AccuracyCase:
    packed: torch.Tensor
    depths: torch.Tensor
    features: torch.Tensor
    target: torch.Tensor
    background: torch.Tensor
    height: int
    width: int
    splat_count: int
    set_index: int
    seed: int


@dataclass(frozen=True)
class AccuracyResult:
    output: torch.Tensor
    loss: float
    packed_grad: torch.Tensor
    feature_grad: torch.Tensor


@dataclass(frozen=True)
class AccuracyRendererSpec:
    name: str
    run: Callable[[AccuracyCase], AccuracyResult]
    device: torch.device
    dtype: torch.dtype
    available: bool = True
    skip_reason: str = ""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"Resolution must be positive, got {value}.")
        return value, value
    if isinstance(value, str):
        if "x" in value:
            left, right = value.lower().split("x", 1)
            return normalize_resolution([int(left), int(right)])
        return normalize_resolution(int(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
        if height < 1 or width < 1:
            raise ValueError(f"Resolution must be positive, got {height}x{width}.")
        return height, width
    raise ValueError(f"Expected resolution as int, 'HxW', or [height, width], got {value!r}.")


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_resolutions(value: str) -> list[tuple[int, int]]:
    return [normalize_resolution(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype: {name}") from exc


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_torch_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def resolve_output_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def safe_filename_part(value: Any) -> str:
    text = str(value).strip().replace(" ", "_")
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


def make_accuracy_case(
    height: int,
    width: int,
    splat_count: int,
    set_index: int,
    seed: int,
    cfg: dict[str, Any],
) -> AccuracyCase:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    margin_x = float(width) * float(cfg["xy_margin_fraction"])
    margin_y = float(height) * float(cfg["xy_margin_fraction"])
    mean_x = torch.rand(splat_count, generator=generator) * max(float(width) - 2.0 * margin_x, 1.0) + margin_x
    mean_y = torch.rand(splat_count, generator=generator) * max(float(height) - 2.0 * margin_y, 1.0) + margin_y
    means = torch.stack([mean_x, mean_y], dim=-1)

    angles = torch.rand(splat_count, generator=generator) * (2.0 * math.pi)
    axes = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    min_sigma, max_sigma = [float(value) for value in cfg["sigma_pixel_range"]]
    sigmas = torch.rand(splat_count, 2, generator=generator) * (max_sigma - min_sigma) + min_sigma

    min_opacity, max_opacity = [float(value) for value in cfg["opacity_range"]]
    alphas = torch.rand(splat_count, 1, generator=generator) * (max_opacity - min_opacity) + min_opacity

    min_feature, max_feature = [float(value) for value in cfg["feature_range"]]
    features = torch.rand(splat_count, 3, generator=generator) * (max_feature - min_feature) + min_feature

    # Unique but shuffled depths keep sort order deterministic across float32/float64.
    depths = torch.linspace(0.05, 0.95, splat_count).view(-1, 1)
    depths = depths[torch.randperm(splat_count, generator=generator)]

    target = torch.rand(3, height, width, generator=generator) * 0.8 + 0.1
    background = torch.tensor(cfg["background"], dtype=torch.float32)

    packed = torch.cat([means, axes, sigmas, alphas], dim=-1)
    return AccuracyCase(
        packed=packed,
        depths=depths,
        features=features,
        target=target,
        background=background,
        height=height,
        width=width,
        splat_count=splat_count,
        set_index=set_index,
        seed=seed,
    )


def clone_for_device(case: AccuracyCase, device: torch.device, dtype: torch.dtype, requires_grad: bool) -> AccuracyCase:
    def tensor(value: torch.Tensor, grad: bool = False) -> torch.Tensor:
        cloned = value.to(device=device, dtype=dtype).detach().clone()
        if grad:
            cloned.requires_grad_(True)
        return cloned

    return AccuracyCase(
        packed=tensor(case.packed, requires_grad),
        depths=tensor(case.depths),
        features=tensor(case.features, requires_grad),
        target=tensor(case.target),
        background=tensor(case.background),
        height=case.height,
        width=case.width,
        splat_count=case.splat_count,
        set_index=case.set_index,
        seed=case.seed,
    )


def torch_reference_rasterize(case: AccuracyCase, cfg: dict[str, Any]) -> torch.Tensor:
    if not bool(cfg.get("use_alpha_blending", True)):
        raise ValueError("The Torch accuracy baseline only implements alpha blending.")
    if bool(cfg.get("antialias", False)):
        raise ValueError("The Torch accuracy baseline only implements the non-antialiased Taichi PDF.")

    packed = case.packed
    features = case.features
    dtype = packed.dtype
    device = packed.device

    y, x = torch.meshgrid(
        torch.arange(case.height, device=device, dtype=dtype),
        torch.arange(case.width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack([x + 0.5, y + 0.5], dim=-1)

    order = torch.argsort(case.depths.squeeze(-1))
    accum = torch.zeros(case.height, case.width, features.shape[-1], device=device, dtype=dtype)
    total_weight = torch.zeros(case.height, case.width, device=device, dtype=dtype)
    alpha_threshold = torch.tensor(float(cfg["alpha_threshold"]), device=device, dtype=dtype)
    clamp_max_alpha = torch.tensor(float(cfg["clamp_max_alpha"]), device=device, dtype=dtype)
    saturate_threshold = torch.tensor(float(cfg["saturate_threshold"]), device=device, dtype=dtype)

    for point_idx in order.tolist():
        point = packed[point_idx]
        mean = point[0:2]
        axis = point[2:4]
        sigma = point[4:6]
        point_alpha = point[6]
        delta = pixels - mean
        perp_axis = torch.stack([-axis[1], axis[0]])
        tx = (delta * axis).sum(dim=-1) / sigma[0]
        ty = (delta * perp_axis).sum(dim=-1) / sigma[1]
        gaussian_alpha = torch.exp(-0.5 * (tx.square() + ty.square()))
        alpha = torch.minimum(point_alpha * gaussian_alpha, clamp_max_alpha)
        active = (alpha > alpha_threshold) & (total_weight < saturate_threshold)
        weight = torch.where(active, alpha * (1.0 - total_weight), torch.zeros_like(alpha))
        accum = accum + weight.unsqueeze(-1) * features[point_idx]
        total_weight = total_weight + weight

    image = accum + (1.0 - total_weight).unsqueeze(-1) * case.background
    return image.permute(2, 0, 1).contiguous()


def build_taichi_rasterizer(device: torch.device, taichi_cfg: dict[str, Any]):
    if device.type not in {"cuda", "mps", "cpu"}:
        raise RuntimeError(f"Unsupported Taichi device: {device}")
    import taichi as ti
    from taichi_splatting.data_types import RasterConfig
    from taichi_splatting.rasterizer import rasterize
    from taichi_splatting.taichi_queue import TaichiQueue

    arch = {"cuda": ti.cuda, "mps": ti.metal, "cpu": ti.cpu}[device.type]
    TaichiQueue.init(arch=arch, log_level=ti.ERROR)
    variant = str(taichi_cfg.get("variant", "metal_reference" if device.type != "cuda" else "cuda_simt"))
    use_depth16 = bool(taichi_cfg.get("use_depth16", False))
    raster_config = RasterConfig(
        tile_size=int(taichi_cfg["tile_size"]),
        alpha_threshold=float(taichi_cfg["alpha_threshold"]),
        clamp_max_alpha=float(taichi_cfg["clamp_max_alpha"]),
        saturate_threshold=float(taichi_cfg["saturate_threshold"]),
        metal_compatible=bool(taichi_cfg.get("metal_compatible", device.type != "cuda")),
        kernel_variant=variant,
        sort_backend=str(taichi_cfg.get("sort_backend", "auto")),
        backward_variant=str(taichi_cfg.get("backward_variant", "pixel_reference")),
        metal_block_dim=int(taichi_cfg.get("metal_block_dim", 0)),
    )

    def render(case: AccuracyCase) -> torch.Tensor:
        raster = rasterize(
            case.packed.contiguous(),
            case.depths.contiguous(),
            case.features.contiguous(),
            image_size=(case.width, case.height),
            config=raster_config,
            use_depth16=use_depth16,
        )
        image = raster.image + (1.0 - raster.image_weight).unsqueeze(-1).clamp(0.0, 1.0) * case.background
        return image.permute(2, 0, 1).contiguous()

    def sync() -> None:
        sync_torch_device(device)
        TaichiQueue.run_sync(ti.sync)
        sync_torch_device(device)

    depth_part = "_depth16" if use_depth16 else ""
    device_part = "metal" if device.type == "mps" else device.type
    sort_backend = str(taichi_cfg.get("sort_backend", "auto"))
    if sort_backend == "taichi_field":
        renderer_name = f"taichi_{device_part}_global_sort{depth_part}"
    elif sort_backend == "bucket_taichi":
        renderer_name = f"taichi_{device_part}_bucket_sort{depth_part}"
    elif sort_backend == "ordered_taichi":
        renderer_name = f"taichi_{device_part}_ordered{depth_part}"
    else:
        renderer_name = f"taichi_{device_part}_{variant.replace('metal_', '').replace('_simt', '')}{depth_part}"
    return render, sync, renderer_name


def build_taichi_accuracy_renderer(
    device: torch.device, dtype: torch.dtype, taichi_cfg: dict[str, Any]
) -> AccuracyRendererSpec:
    try:
        render_taichi, sync_taichi, renderer_name = build_taichi_rasterizer(device, taichi_cfg)
    except Exception as exc:
        return AccuracyRendererSpec(
            "taichi",
            run=lambda _case: AccuracyResult(torch.empty(0), float("nan"), torch.empty(0), torch.empty(0)),
            device=device,
            dtype=dtype,
            available=False,
            skip_reason=str(exc),
        )

    def run(case: AccuracyCase) -> AccuracyResult:
        output = render_taichi(case)
        sync_taichi()
        loss = mse_loss(output, case.target)
        loss.backward()
        sync_taichi()
        if case.packed.grad is None or case.features.grad is None:
            raise RuntimeError("Taichi accuracy renderer did not produce packed/features gradients.")
        return AccuracyResult(
            output=output.detach(),
            loss=float(loss.detach().cpu().item()),
            packed_grad=case.packed.grad.detach(),
            feature_grad=case.features.grad.detach(),
        )

    return AccuracyRendererSpec(renderer_name, run=run, device=device, dtype=dtype)


def build_raw_metal_accuracy_renderer(raw_metal_cfg: dict[str, Any]) -> AccuracyRendererSpec:
    renderer_name = "raw_metal_mlx"
    try:
        import_raw_metal()
    except RawMetalUnavailable as exc:
        return AccuracyRendererSpec(
            renderer_name,
            run=lambda _case: AccuracyResult(torch.empty(0), float("nan"), torch.empty(0), torch.empty(0)),
            device=torch.device("cpu"),
            dtype=torch.float32,
            available=False,
            skip_reason=str(exc),
        )

    def run(case: AccuracyCase) -> AccuracyResult:
        output, packed_grad, feature_grad, loss = run_packed_accuracy(
            case.packed,
            case.depths,
            case.features,
            case.target,
            case.background,
            case.height,
            case.width,
            raw_metal_cfg,
        )
        return AccuracyResult(
            output=output,
            loss=loss,
            packed_grad=packed_grad,
            feature_grad=feature_grad,
        )

    return AccuracyRendererSpec(renderer_name, run=run, device=torch.device("cpu"), dtype=torch.float32)


def build_accuracy_renderers(
    device: torch.device,
    dtype: torch.dtype,
    config: dict[str, Any],
    requested_keys: list[str] | None = None,
) -> dict[str, AccuracyRendererSpec]:
    requested = set(requested_keys) if requested_keys is not None else {"taichi", "raw_metal"}
    specs: dict[str, AccuracyRendererSpec] = {}
    if "taichi" in requested:
        specs["taichi"] = build_taichi_accuracy_renderer(device, dtype, config["taichi"])
    if "raw_metal" in requested:
        specs["raw_metal"] = build_raw_metal_accuracy_renderer(config["raw_metal"])
    return specs


def mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (output - target).square().mean()


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor, prefix: str) -> dict[str, float]:
    candidate_cpu = candidate.detach().cpu().to(dtype=torch.float64)
    reference_cpu = reference.detach().cpu().to(dtype=torch.float64)
    diff = (candidate_cpu - reference_cpu).abs()
    reference_abs = reference_cpu.abs()
    return {
        f"{prefix}_max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        f"{prefix}_mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        f"{prefix}_max_rel": float((diff / reference_abs.clamp_min(1.0e-8)).max().item()) if diff.numel() else 0.0,
    }


def within_tolerance(row: dict[str, Any], tolerances: dict[str, Any]) -> bool:
    checks = {
        "image_max_abs": row["image_max_abs"],
        "image_mean_abs": row["image_mean_abs"],
        "loss_abs": row["loss_abs"],
        "packed_grad_max_abs": row["packed_grad_max_abs"],
        "packed_grad_mean_abs": row["packed_grad_mean_abs"],
        "feature_grad_max_abs": row["feature_grad_max_abs"],
        "feature_grad_mean_abs": row["feature_grad_mean_abs"],
    }
    return all(value <= float(tolerances[name]) for name, value in checks.items())


def save_chw_image(output: torch.Tensor, path: Path) -> None:
    from PIL import Image

    image = output.detach()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {tuple(image.shape)}.")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] > 3:
        image = image[:3]
    array = image.clamp(0.0, 1.0).nan_to_num(0.0).permute(1, 2, 0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_case_images(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    row: dict[str, Any],
    directory: Path,
) -> list[str]:
    stem = f"{int(row['height'])}x{int(row['width'])}__G{int(row['splat_count'])}__set{int(row['set_index'])}"
    ref_path = directory / f"torch_reference__{stem}.png"
    taichi_path = directory / f"{safe_filename_part(row['renderer'])}__{stem}.png"
    diff_path = directory / f"abs_diff_x20__{safe_filename_part(row['renderer'])}__{stem}.png"
    save_chw_image(reference, ref_path)
    save_chw_image(candidate, taichi_path)
    save_chw_image((candidate.detach().cpu() - reference.detach().cpu()).abs() * 20.0, diff_path)
    return [str(ref_path), str(taichi_path), str(diff_path)]


def image_save_target(
    config: dict[str, Any], resolutions: list[tuple[int, int]], splat_counts: list[int]
) -> tuple[set[tuple[int, int]], set[int], int]:
    save_config = config.get("save_images", {})
    if bool(save_config.get("largest_resolution_only", True)):
        max_area = max(height * width for height, width in resolutions)
        target_resolutions = {(height, width) for height, width in resolutions if height * width == max_area}
    else:
        target_resolutions = set(resolutions)

    if bool(save_config.get("largest_splat_count_only", True)):
        target_splat_counts = {max(splat_counts)}
    else:
        target_splat_counts = set(splat_counts)

    return target_resolutions, target_splat_counts, int(save_config.get("set_index", 0))


def should_save_image(
    row: dict[str, Any], target_resolutions: set[tuple[int, int]], target_splat_counts: set[int], target_set_index: int
) -> bool:
    return (
        (int(row["height"]), int(row["width"])) in target_resolutions
        and int(row["splat_count"]) in target_splat_counts
        and int(row["set_index"]) == target_set_index
    )


def run_accuracy(config: dict[str, Any], fail_on_mismatch: bool) -> list[dict[str, Any]]:
    device = pick_device(str(config["device"]))
    dtype = dtype_from_name(str(config["dtype"]))
    baseline_device = pick_device(str(config["baseline_device"]))
    baseline_dtype = dtype_from_name(str(config["baseline_dtype"]))
    resolutions = [normalize_resolution(value) for value in config["resolutions"]]
    splat_counts = [int(value) for value in config["splat_counts"]]
    requested_renderers = [str(name) for name in config["renderers"]]
    renderers = build_accuracy_renderers(device, dtype, config, requested_renderers)
    unknown_renderers = [name for name in requested_renderers if name not in renderers]
    if unknown_renderers:
        raise ValueError(f"Unknown renderer(s): {', '.join(unknown_renderers)}")

    save_config = config.get("save_images", {})
    save_images = bool(save_config.get("enabled", False))
    image_directory = resolve_output_path(save_config.get("directory", "benchmark_outputs/splat_accuracy_images"))
    target_resolutions, target_splat_counts, target_set_index = image_save_target(config, resolutions, splat_counts)

    print(f"device={device} dtype={dtype} baseline_device={baseline_device} baseline_dtype={baseline_dtype}")
    print(f"renderers={requested_renderers}")
    print(f"resolutions={resolutions} splat_counts={splat_counts} sets_per_case={config['sets_per_case']}")
    print("")

    rows: list[dict[str, Any]] = []
    base_seed = int(config["seed"])
    for height, width in resolutions:
        for splat_count in splat_counts:
            for set_index in range(int(config["sets_per_case"])):
                seed = base_seed + height * 1_000_003 + width * 10_007 + splat_count * 101 + set_index
                base_case = make_accuracy_case(height, width, splat_count, set_index, seed, config["random_splats_2d"])

                reference_case = clone_for_device(base_case, baseline_device, baseline_dtype, requires_grad=True)
                reference_output = torch_reference_rasterize(reference_case, config["taichi"])
                reference_loss = mse_loss(reference_output, reference_case.target)
                reference_loss.backward()

                for renderer_key in requested_renderers:
                    renderer = renderers[renderer_key]
                    row: dict[str, Any] = {
                        "renderer": renderer.name,
                        "renderer_key": renderer_key,
                        "height": height,
                        "width": width,
                        "splat_count": splat_count,
                        "set_index": set_index,
                        "seed": seed,
                        "renderer_device": str(renderer.device),
                        "baseline_device": str(baseline_device),
                        "renderer_dtype": str(renderer.dtype).replace("torch.", ""),
                        "baseline_dtype": str(baseline_dtype).replace("torch.", ""),
                        "reference_loss": float(reference_loss.detach().cpu().item()),
                    }
                    if not renderer.available:
                        row.update({"status": "skipped", "passed": False, "skip_reason": renderer.skip_reason})
                        rows.append(row)
                        print(
                            f"SKIP {row['renderer']:<14} {height:>4}x{width:<4} G={splat_count:<4} set={set_index:<2} "
                            f"{row['skip_reason']}"
                        )
                        continue

                    try:
                        candidate_case = clone_for_device(
                            base_case, renderer.device, renderer.dtype, requires_grad=True
                        )
                        result = renderer.run(candidate_case)
                    except Exception as exc:
                        row.update({"status": "error", "passed": False, "error": repr(exc)})
                        rows.append(row)
                        print(
                            f"ERR  {row['renderer']:<14} {height:>4}x{width:<4} G={splat_count:<4} set={set_index:<2} "
                            f"{row['error']}"
                        )
                        continue

                    row["status"] = "ok"
                    row["candidate_loss"] = result.loss
                    row["loss_abs"] = abs(result.loss - row["reference_loss"])
                    row.update(compare_tensors(reference_output, result.output, "image"))
                    row.update(compare_tensors(reference_case.packed.grad, result.packed_grad, "packed_grad"))
                    row.update(compare_tensors(reference_case.features.grad, result.feature_grad, "feature_grad"))
                    row["passed"] = within_tolerance(row, config["tolerances"])

                    if save_images and should_save_image(
                        row, target_resolutions, target_splat_counts, target_set_index
                    ):
                        row["saved_image_paths"] = save_case_images(
                            reference_output.cpu(), result.output.cpu(), row, image_directory
                        )

                    rows.append(row)
                    status = "PASS" if row["passed"] else "FAIL"
                    print(
                        f"{status:<4} {row['renderer']:<14} {height:>4}x{width:<4} G={splat_count:<4} set={set_index:<2} "
                        f"img_max={row['image_max_abs']:.3e} img_mean={row['image_mean_abs']:.3e} "
                        f"loss={row['loss_abs']:.3e} "
                        f"packed_grad_max={row['packed_grad_max_abs']:.3e} feature_grad_max={row['feature_grad_max_abs']:.3e}"
                    )

    print_summary(rows)
    if fail_on_mismatch and any(row["status"] != "skipped" and not row["passed"] for row in rows):
        raise SystemExit(1)
    return rows


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    skipped_rows = [row for row in rows if row.get("status") == "skipped"]
    error_rows = [row for row in rows if row.get("status") == "error"]

    print("\nSummary max errors by resolution/splats:")
    key_fn = lambda row: (row["height"], row["width"], row["splat_count"])
    for key, group_iter in itertools.groupby(sorted(ok_rows, key=key_fn), key=key_fn):
        group = list(group_iter)
        height, width, splat_count = key
        passed = sum(1 for row in group if row["passed"])
        print(
            f"{height:>4}x{width:<4} G={splat_count:<4} passed={passed}/{len(group)} "
            f"img_max={max(row['image_max_abs'] for row in group):.3e} "
            f"img_mean={max(row['image_mean_abs'] for row in group):.3e} "
            f"packed_grad_max={max(row['packed_grad_max_abs'] for row in group):.3e} "
            f"feature_grad_max={max(row['feature_grad_max_abs'] for row in group):.3e}"
        )

    if skipped_rows:
        print("\nSummary skipped renderer cases:")
        skip_key_fn = lambda row: (row["renderer"], row["height"], row["width"], row["splat_count"], row["skip_reason"])
        for key, group_iter in itertools.groupby(sorted(skipped_rows, key=skip_key_fn), key=skip_key_fn):
            group = list(group_iter)
            renderer, height, width, splat_count, reason = key
            print(f"{renderer:<14} {height:>4}x{width:<4} G={splat_count:<4} skipped={len(group):<3} reason={reason}")

    if error_rows:
        print("\nSummary errored renderer cases:")
        error_key_fn = lambda row: (row["renderer"], row["height"], row["width"], row["splat_count"])
        for key, group_iter in itertools.groupby(sorted(error_rows, key=error_key_fn), key=error_key_fn):
            group = list(group_iter)
            renderer, height, width, splat_count = key
            print(f"{renderer:<14} {height:>4}x{width:<4} G={splat_count:<4} errors={len(group)}")

    saved_paths = [path for row in ok_rows for path in row.get("saved_image_paths", [])]
    if saved_paths:
        print("\nSaved accuracy images:")
        for path in saved_paths:
            print(path)


def load_accuracy_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    return deep_merge(DEFAULT_CONFIG, load_config_file(path))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Taichi splat rasterization against a direct Torch baseline.")
    parser.add_argument("--config", type=Path, help="Optional JSONC accuracy config.")
    parser.add_argument("--device", type=str, help="Override Taichi device, e.g. mps, cuda, cpu.")
    parser.add_argument("--baseline-device", type=str, help="Override Torch baseline device.")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], help="Override Taichi dtype.")
    parser.add_argument(
        "--baseline-dtype", type=str, choices=["float32", "float64"], help="Override Torch baseline dtype."
    )
    parser.add_argument("--renderers", type=str, help="Comma-separated renderer keys.")
    parser.add_argument("--resolutions", type=str, help="Comma-separated sizes, e.g. 16,32x24.")
    parser.add_argument("--splat-counts", type=str, help="Comma-separated splat counts.")
    parser.add_argument("--sets-per-case", type=int, help="Random cases per resolution/count combination.")
    parser.add_argument("--save-images", type=Path, help="Save selected reference/Taichi/diff PNGs to this directory.")
    parser.add_argument("--no-save-images", action="store_true", help="Disable accuracy PNG output.")
    parser.add_argument("--fail-on-mismatch", action="store_true", help="Exit nonzero if any row exceeds tolerance.")
    return parser.parse_args(argv)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(config)
    if args.device is not None:
        cfg["device"] = args.device
    if args.baseline_device is not None:
        cfg["baseline_device"] = args.baseline_device
    if args.dtype is not None:
        cfg["dtype"] = args.dtype
    if args.baseline_dtype is not None:
        cfg["baseline_dtype"] = args.baseline_dtype
    if args.renderers is not None:
        cfg["renderers"] = parse_csv_strings(args.renderers)
    if args.resolutions is not None:
        cfg["resolutions"] = parse_csv_resolutions(args.resolutions)
    if args.splat_counts is not None:
        cfg["splat_counts"] = parse_csv_ints(args.splat_counts)
    if args.sets_per_case is not None:
        cfg["sets_per_case"] = args.sets_per_case
    if args.save_images is not None:
        cfg.setdefault("save_images", {})
        cfg["save_images"]["enabled"] = True
        cfg["save_images"]["directory"] = str(args.save_images)
    if args.no_save_images:
        cfg.setdefault("save_images", {})
        cfg["save_images"]["enabled"] = False
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_cli_overrides(load_accuracy_config(args.config), args)
    run_accuracy(config, fail_on_mismatch=bool(args.fail_on_mismatch))


if __name__ == "__main__":
    main()
