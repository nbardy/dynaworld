from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


GAUSSIAN_FIELD_NAMES = ("xyz", "scales", "quats", "opacities", "rgbs")
GAUSSIAN_HEAD_CHANNELS = {
    "xyz_raw": 3,
    "scale_raw": 3,
    "rot_raw": 4,
    "opacity_raw": 1,
    "rgb_raw": 3,
}


def _finite_flattened(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.detach().float().reshape(-1)
    return values[torch.isfinite(values)]


def _float_or_nan(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return float("nan")
    return float(value.detach().cpu())


def _skewness(tensor: torch.Tensor) -> float:
    finite = _finite_flattened(tensor)
    if finite.numel() == 0:
        return float("nan")
    centered = finite - finite.mean()
    std = finite.std(unbiased=False)
    if float(std.detach().cpu()) <= 1.0e-12:
        return 0.0
    return _float_or_nan((centered / std).pow(3).mean())


def _quantile(tensor: torch.Tensor, value: float) -> float:
    finite = _finite_flattened(tensor)
    if finite.numel() == 0:
        return float("nan")
    return _float_or_nan(torch.quantile(finite.cpu(), value))


def histogram_entropy01(
    tensor: torch.Tensor,
    *,
    lower: float,
    upper: float,
    bins: int = 20,
) -> float:
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}.")
    if upper <= lower:
        return float("nan")
    finite = _finite_flattened(tensor)
    if finite.numel() == 0:
        return float("nan")
    histogram = torch.histc(finite.clamp(lower, upper).cpu(), bins=bins, min=lower, max=upper)
    total = histogram.sum()
    if float(total) <= 0.0:
        return float("nan")
    probabilities = histogram / total
    probabilities = probabilities[probabilities > 0]
    if probabilities.numel() == 0:
        return float("nan")
    return _float_or_nan(-(probabilities * probabilities.log()).sum() / math.log(bins))


def edge_fraction(
    tensor: torch.Tensor,
    *,
    lower: float,
    upper: float,
    edge_fraction_width: float = 0.05,
) -> float:
    if upper <= lower:
        return float("nan")
    finite = _finite_flattened(tensor)
    if finite.numel() == 0:
        return float("nan")
    margin = (upper - lower) * float(edge_fraction_width)
    near_edge = (finite <= lower + margin) | (finite >= upper - margin)
    return _float_or_nan(near_edge.float().mean())


def distribution_stats(
    tensor: torch.Tensor,
    *,
    lower: float | None = None,
    upper: float | None = None,
    bins: int = 20,
) -> dict[str, float]:
    finite = _finite_flattened(tensor)
    metrics = {
        "Min": _float_or_nan(finite.min()) if finite.numel() else float("nan"),
        "Max": _float_or_nan(finite.max()) if finite.numel() else float("nan"),
        "Mean": _float_or_nan(finite.mean()) if finite.numel() else float("nan"),
        "Std": _float_or_nan(finite.std(unbiased=False)) if finite.numel() else float("nan"),
        "Skew": _skewness(tensor),
        "P01": _quantile(tensor, 0.01),
        "P50": _quantile(tensor, 0.50),
        "P99": _quantile(tensor, 0.99),
        "NonfiniteCount": float((~torch.isfinite(tensor.detach())).sum().detach().cpu()),
    }
    if lower is not None and upper is not None:
        span = float(upper) - float(lower)
        metrics["Coverage"] = (metrics["Max"] - metrics["Min"]) / span if span > 0 else float("nan")
        metrics["Entropy01"] = histogram_entropy01(tensor, lower=float(lower), upper=float(upper), bins=bins)
        metrics["Edge05"] = edge_fraction(tensor, lower=float(lower), upper=float(upper), edge_fraction_width=0.05)
    return metrics


def _add_prefixed_metrics(payload: dict[str, float], prefix: str, stats: Mapping[str, float]) -> None:
    for key, value in stats.items():
        payload[f"{prefix}/{key}"] = float(value)


def _field_tensor(decoded: Any, field_name: str) -> torch.Tensor:
    if isinstance(decoded, Mapping):
        value = decoded[field_name]
    else:
        value = getattr(decoded, field_name)
    if not torch.is_tensor(value):
        raise TypeError(f"Decoded field {field_name!r} must be a tensor, got {type(value).__name__}.")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"Decoded field {field_name!r} must have shape [F,G,C] or [G,C], got {tuple(value.shape)}.")
    return value


def _spread_stats(
    tensor: torch.Tensor,
    *,
    token_count: int,
    gaussians_per_token: int,
) -> dict[str, float]:
    if token_count < 1:
        raise ValueError(f"token_count must be >= 1, got {token_count}.")
    if gaussians_per_token < 1:
        raise ValueError(f"gaussians_per_token must be >= 1, got {gaussians_per_token}.")

    value = tensor.detach().float()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    expected_gaussians = int(token_count) * int(gaussians_per_token)
    if value.shape[1] != expected_gaussians:
        raise ValueError(
            f"Expected {expected_gaussians} gaussians from token_count={token_count} and "
            f"gaussians_per_token={gaussians_per_token}, got {value.shape[1]}."
        )

    shaped = value.reshape(value.shape[0], token_count, gaussians_per_token, value.shape[-1])
    within_token_std = shaped.std(dim=2, unbiased=False).mean()
    within_token_range = (shaped.max(dim=2).values - shaped.min(dim=2).values).mean()
    same_split_cross_token_std = shaped.std(dim=1, unbiased=False).mean()
    token_centroid_std = shaped.mean(dim=2).std(dim=1, unbiased=False).mean()
    flat_std = shaped.reshape(-1, shaped.shape[-1]).std(dim=0, unbiased=False).mean()
    ratio = same_split_cross_token_std / within_token_std.clamp_min(1.0e-12)
    return {
        "FlatStdMean": _float_or_nan(flat_std),
        "WithinTokenStdMean": _float_or_nan(within_token_std),
        "WithinTokenRangeMean": _float_or_nan(within_token_range),
        "SameSplitCrossTokenStdMean": _float_or_nan(same_split_cross_token_std),
        "TokenCentroidStdMean": _float_or_nan(token_centroid_std),
        "CrossToWithinStdRatio": _float_or_nan(ratio),
    }


def infer_valid_ranges_from_config(config: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    model_cfg = config.get("model", {})
    arch = config.get("arch")
    ranges: dict[str, tuple[float, float]] = {
        "opacities": (0.0, 1.0),
        "rgbs": (0.0, 1.0),
    }
    if "xy_extent" in model_cfg:
        xy_extent = float(model_cfg["xy_extent"])
        ranges["xyz/x"] = (-xy_extent, xy_extent)
        ranges["xyz/y"] = (-xy_extent, xy_extent)
    elif "scene_extent" in model_cfg:
        scene_extent = float(model_cfg["scene_extent"])
        ranges["xyz/x"] = (-scene_extent, scene_extent)
        ranges["xyz/y"] = (-scene_extent, scene_extent)
        ranges["xyz/z"] = (-scene_extent, scene_extent)

    if "z_min" in model_cfg and "z_max" in model_cfg:
        ranges["xyz/z"] = (float(model_cfg["z_min"]), float(model_cfg["z_max"]))
    elif "scene_extent" in model_cfg:
        scene_extent = float(model_cfg["scene_extent"])
        ranges["xyz/z"] = (-scene_extent, scene_extent)
    elif arch == "tokengs_image_implicit_camera":
        scene_extent = 1.0
        ranges["xyz/x"] = (-scene_extent, scene_extent)
        ranges["xyz/y"] = (-scene_extent, scene_extent)
        ranges["xyz/z"] = (-scene_extent, scene_extent)
    return ranges


def decoded_gaussian_init_diagnostics(
    decoded: Any,
    *,
    token_count: int,
    gaussians_per_token: int,
    valid_ranges: Mapping[str, tuple[float, float]] | None = None,
    prefix: str = "InitDiag",
    bins: int = 20,
) -> dict[str, float]:
    valid_ranges = valid_ranges or {}
    metrics: dict[str, float] = {}

    xyz = _field_tensor(decoded, "xyz")
    for axis, axis_name in enumerate(("x", "y", "z")):
        range_key = f"xyz/{axis_name}"
        lower_upper = valid_ranges.get(range_key)
        stats = distribution_stats(
            xyz[..., axis],
            lower=lower_upper[0] if lower_upper is not None else None,
            upper=lower_upper[1] if lower_upper is not None else None,
            bins=bins,
        )
        _add_prefixed_metrics(metrics, f"{prefix}/XYZ/{axis_name.upper()}", stats)
    _add_prefixed_metrics(
        metrics,
        f"{prefix}/Spread/XYZ",
        _spread_stats(xyz, token_count=token_count, gaussians_per_token=gaussians_per_token),
    )

    scales = _field_tensor(decoded, "scales")
    _add_prefixed_metrics(metrics, f"{prefix}/Scale", distribution_stats(scales, bins=bins))
    positive_scales = torch.where(scales > 0, scales, torch.full_like(scales, float("nan")))
    _add_prefixed_metrics(metrics, f"{prefix}/LogScale", distribution_stats(torch.log(positive_scales), bins=bins))
    _add_prefixed_metrics(
        metrics,
        f"{prefix}/Spread/Scale",
        _spread_stats(scales, token_count=token_count, gaussians_per_token=gaussians_per_token),
    )

    quats = _field_tensor(decoded, "quats")
    _add_prefixed_metrics(metrics, f"{prefix}/QuatNorm", distribution_stats(quats.norm(dim=-1), bins=bins))

    for field_name, display_name in (("opacities", "Opacity"), ("rgbs", "RGB")):
        tensor = _field_tensor(decoded, field_name)
        lower_upper = valid_ranges.get(field_name)
        stats = distribution_stats(
            tensor,
            lower=lower_upper[0] if lower_upper is not None else None,
            upper=lower_upper[1] if lower_upper is not None else None,
            bins=bins,
        )
        _add_prefixed_metrics(metrics, f"{prefix}/{display_name}", stats)
        _add_prefixed_metrics(
            metrics,
            f"{prefix}/Spread/{display_name}",
            _spread_stats(tensor, token_count=token_count, gaussians_per_token=gaussians_per_token),
        )
    return metrics


def gaussian_head_raw_outputs(gaussian_heads: Any, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    head_specs = {
        "xyz_raw": "xyz_head",
        "scale_raw": "scale_head",
        "rot_raw": "rot_head",
        "opacity_raw": "opacity_head",
        "rgb_raw": "rgb_head",
    }
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens with shape [B,T,C], got {tuple(tokens.shape)}.")
    batch_size, token_count, _channels = tokens.shape
    outputs = {}
    for output_name, module_name in head_specs.items():
        if not hasattr(gaussian_heads, module_name):
            continue
        raw = getattr(gaussian_heads, module_name)(tokens)
        channels = GAUSSIAN_HEAD_CHANNELS[output_name]
        outputs[output_name] = raw.reshape(batch_size, token_count * gaussian_heads.gaussians_per_token, channels)
    return outputs


def raw_head_output_diagnostics(
    raw_outputs: Mapping[str, torch.Tensor],
    *,
    prefix: str = "InitRaw",
    bins: int = 20,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, tensor in raw_outputs.items():
        _add_prefixed_metrics(metrics, f"{prefix}/{name}", distribution_stats(tensor, bins=bins))
        metrics[f"{prefix}/{name}/AbsGt3Fraction"] = _float_or_nan((_finite_flattened(tensor).abs() > 3.0).float().mean())
        metrics[f"{prefix}/{name}/AbsGt5Fraction"] = _float_or_nan((_finite_flattened(tensor).abs() > 5.0).float().mean())
    return metrics


def format_init_diagnostic_summary(metrics: Mapping[str, float]) -> str:
    keys = (
        "InitDiag/XYZ/X/Coverage",
        "InitDiag/XYZ/Y/Coverage",
        "InitDiag/XYZ/Z/Coverage",
        "InitDiag/XYZ/X/Entropy01",
        "InitDiag/XYZ/Y/Entropy01",
        "InitDiag/XYZ/Z/Entropy01",
        "InitDiag/Spread/XYZ/WithinTokenStdMean",
        "InitDiag/Spread/XYZ/SameSplitCrossTokenStdMean",
        "InitDiag/Spread/XYZ/CrossToWithinStdRatio",
        "InitDiag/Opacity/Min",
        "InitDiag/Opacity/Max",
        "InitDiag/RGB/Std",
        "InitDiag/QuatNorm/Min",
        "InitDiag/QuatNorm/Mean",
        "InitDiag/QuatNorm/Max",
    )
    return ", ".join(f"{key}={metrics[key]:.4g}" for key in keys if key in metrics)
