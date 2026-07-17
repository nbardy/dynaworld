from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

try:
    from .report_artifacts import split_csv_floats, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import split_csv_floats, write_report_json, write_report_text
from objective.objective import colorize_and_compose_feature_rgb
from objective.types import BackgroundSample


DEFAULT_ALPHAS = (0.0, 0.02, 0.1, 0.5, 1.0)


def _make_colorizer() -> torch.nn.Conv2d:
    colorizer = torch.nn.Conv2d(2, 3, kernel_size=1, bias=True)
    with torch.no_grad():
        colorizer.weight.zero_()
        colorizer.weight[0, 0, 0, 0] = 1.0
        colorizer.weight[1, 1, 0, 0] = 1.0
        colorizer.weight[2, 0, 0, 0] = 0.5
        colorizer.weight[2, 1, 0, 0] = -0.5
        colorizer.bias[:] = torch.tensor([0.05, -0.03, 0.02])
    return colorizer


def _grad_l2(parameters: torch.nn.Module) -> float:
    total = 0.0
    for param in parameters.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().square().sum().item())
    return math.sqrt(total)


def _base_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature = torch.tensor([[[[0.30]], [[-0.20]]]], dtype=torch.float32)
    rgb_background = torch.tensor([[[[0.10]], [[0.90]], [[0.70]]]], dtype=torch.float32)
    feature_background = torch.tensor([[[[-0.70]], [[0.60]]]], dtype=torch.float32)
    target_rgb = torch.tensor([[[[0.80]], [[0.20]], [[0.40]]]], dtype=torch.float32)
    return feature, rgb_background, feature_background, target_rgb


def rgb_background_after_colorizer_row(alpha_value: float) -> dict[str, Any]:
    feature, rgb_background, _feature_background, target_rgb = _base_tensors()
    feature = feature.clone().requires_grad_(True)
    alpha = torch.tensor([[[float(alpha_value)]]], dtype=torch.float32, requires_grad=True)
    colorizer = _make_colorizer()

    background = BackgroundSample(rgb=rgb_background, mode="fixed_rgb", phase="train")
    rendered = colorize_and_compose_feature_rgb(feature, alpha, colorizer, background)
    loss = (rendered - target_rgb).square().mean()
    loss.backward()

    return {
        "mode": "rgb_background_after_colorizer",
        "alpha": float(alpha_value),
        "loss": float(loss.detach().item()),
        "feature_grad_l2": float(feature.grad.detach().norm().item()),
        "alpha_grad": float(alpha.grad.detach().item()),
        "colorizer_grad_l2": _grad_l2(colorizer),
    }


def feature_background_before_colorizer_row(alpha_value: float) -> dict[str, Any]:
    feature, _rgb_background, feature_background, target_rgb = _base_tensors()
    feature = feature.clone().requires_grad_(True)
    alpha = torch.tensor([[[float(alpha_value)]]], dtype=torch.float32, requires_grad=True)
    colorizer = _make_colorizer()

    background = BackgroundSample(
        rgb=None,
        mode="none",
        phase="train",
        feature=feature_background,
        feature_mode="fixed_zero",
    )
    rendered = colorize_and_compose_feature_rgb(alpha.unsqueeze(1) * feature, alpha, colorizer, background)
    loss = (rendered - target_rgb).square().mean()
    loss.backward()

    return {
        "mode": "feature_background_before_colorizer",
        "alpha": float(alpha_value),
        "loss": float(loss.detach().item()),
        "feature_grad_l2": float(feature.grad.detach().norm().item()),
        "alpha_grad": float(alpha.grad.detach().item()),
        "colorizer_grad_l2": _grad_l2(colorizer),
    }


def run_diagnostic(alphas: tuple[float, ...] = DEFAULT_ALPHAS) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        rows.append(rgb_background_after_colorizer_row(alpha))
        rows.append(feature_background_before_colorizer_row(alpha))
    return {
        "read": (
            "RGB background after colorizer gates colorizer gradients by alpha. "
            "Feature background before colorizer trains the colorizer even at alpha=0."
        ),
        "rows": rows,
    }


def _format_float(value: float) -> str:
    return f"{float(value):.6g}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# STAR UVT Background Cheat Diagnostic",
        "",
        str(report["read"]),
        "",
        "| mode | alpha | loss | feature grad L2 | alpha grad | colorizer grad L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {mode} | {alpha} | {loss} | {feature_grad_l2} | {alpha_grad} | {colorizer_grad_l2} |".format(
                mode=row["mode"],
                alpha=_format_float(row["alpha"]),
                loss=_format_float(row["loss"]),
                feature_grad_l2=_format_float(row["feature_grad_l2"]),
                alpha_grad=_format_float(row["alpha_grad"]),
                colorizer_grad_l2=_format_float(row["colorizer_grad_l2"]),
            )
        )
    write_report_text(path, "\n".join(lines) + "\n")


def _parse_alphas(raw: str) -> tuple[float, ...]:
    values = split_csv_floats(raw)
    if not values:
        raise argparse.ArgumentTypeError("at least one alpha value is required")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise argparse.ArgumentTypeError("alpha values must be in [0, 1]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", type=_parse_alphas, default=DEFAULT_ALPHAS)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs/benchmarks/2026-05-21_star_uvt_background_cheat_diagnostic.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("outputs/benchmarks/2026-05-21_star_uvt_background_cheat_diagnostic.md"),
    )
    args = parser.parse_args()

    report = run_diagnostic(tuple(args.alphas))
    write_report_json(args.out_json, report)
    write_markdown(report, args.out_md)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
