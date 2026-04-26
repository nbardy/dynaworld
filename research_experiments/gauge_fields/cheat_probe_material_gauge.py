from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
DYNAWORLD_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from train import (  # noqa: E402
    MaterialSurfelField,
    RenderConfig,
    alpha_metrics,
    flow_health_metrics,
    gauge_config,
    load_baseline_video,
    model_metrics,
    motion_health_metrics,
    path_or_none,
    projection_health_metrics,
    render_sequence,
    resolve_device,
    resolve_dynaworld_path,
    save_side_by_side_mp4,
    select_configured_frames,
    tensor_to_uint8_image,
    video_metrics,
    write_json,
    xmap_health_metrics,
)


ProbeFn = Callable[[MaterialSurfelField, argparse.Namespace], MaterialSurfelField]


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def build_model_from_state(state: dict[str, torch.Tensor], device: torch.device) -> MaterialSurfelField:
    x0 = state["x0"].detach().to(device)
    coeff = state["nr_coeff"].detach().to(device)
    num_frames = int(coeff.shape[0])
    num_basis = int(coeff.shape[1])
    model = MaterialSurfelField(
        init_x0=x0,
        num_frames=num_frames,
        num_basis=num_basis,
        init_radius=0.01,
        init_color=None,
        init_alpha_logit=0.0,
    ).to(device)
    model.load_state_dict({key: value.detach().to(device) for key, value in state.items()})
    model.eval()
    return model


def clone_model(model: MaterialSurfelField) -> MaterialSurfelField:
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    return build_model_from_state(state, model.x0.device)


def append_elements(
    model: MaterialSurfelField,
    x0_new: torch.Tensor,
    color_logits_new: torch.Tensor,
    raw_alpha_new: torch.Tensor,
    log_radius_new: torch.Tensor,
    basis_new: torch.Tensor,
) -> MaterialSurfelField:
    state = model.state_dict()
    expanded_state = {
        "x0": torch.cat([state["x0"], x0_new], dim=0),
        "color_logits": torch.cat([state["color_logits"], color_logits_new], dim=0),
        "raw_alpha": torch.cat([state["raw_alpha"], raw_alpha_new], dim=0),
        "log_radius": torch.cat([state["log_radius"], log_radius_new], dim=0),
        "nr_basis": torch.cat([state["nr_basis"], basis_new], dim=0),
        "nr_coeff": state["nr_coeff"].clone(),
    }
    return build_model_from_state(expanded_state, model.x0.device)


def load_target_video(cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    data_cfg = cfg["data"]
    render_cfg = cfg["render"]
    frames_dir = path_or_none(data_cfg["frames_dir"])
    if frames_dir is not None:
        frames_dir = resolve_dynaworld_path(frames_dir)
    video_path = path_or_none(data_cfg["video_path"])
    if video_path is not None:
        video_path = resolve_dynaworld_path(video_path)
    video = load_baseline_video(
        sequence_dir=resolve_dynaworld_path(data_cfg["sequence_dir"]),
        frames_dir=frames_dir,
        video_path=video_path,
        frame_source=str(data_cfg["frame_source"]),
        render_size=int(render_cfg["render_size"]),
        max_frames=int(data_cfg["max_frames"]),
        device=device,
    )
    return select_configured_frames(video, data_cfg["frame_indices"])


def render_config_from_checkpoint(checkpoint: dict[str, Any], cfg: dict[str, Any]) -> RenderConfig:
    if "render_config" in checkpoint:
        return RenderConfig(**checkpoint["render_config"])
    render_cfg = cfg["render"]
    return RenderConfig(
        H=int(render_cfg["render_size"]),
        W=int(render_cfg["render_size"]),
        near=float(render_cfg["near_plane"]),
        far=float(render_cfg["far_plane"]),
        bg=float(render_cfg["background"][0] if isinstance(render_cfg["background"], list) else render_cfg["background"]),
        min_radius_px=float(render_cfg["min_radius_px"]),
        max_radius_px=float(render_cfg["max_radius_px"]),
        max_alpha_per_element=float(render_cfg["max_alpha_per_element"]),
        pixel_chunk=int(render_cfg["pixel_chunk"]),
    )


def choose_indices(model: MaterialSurfelField, sample_fraction: float, seed: int) -> torch.Tensor:
    count = max(1, min(model.N, int(round(model.N * float(sample_fraction)))))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    idx = torch.randperm(model.N, generator=gen)[:count]
    return idx.to(model.x0.device)


@torch.no_grad()
def probe_depth_slide(model: MaterialSurfelField, args: argparse.Namespace) -> MaterialSurfelField:
    idx = choose_indices(model, args.sample_fraction, args.seed)
    direction = model.x0[idx] / model.x0[idx].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    model.x0[idx] += float(args.depth_slide_eps) * direction
    return model


@torch.no_grad()
def probe_radius_inflate(model: MaterialSurfelField, args: argparse.Namespace) -> MaterialSurfelField:
    idx = choose_indices(model, args.sample_fraction, args.seed)
    model.log_radius[idx] += float(args.radius_log_scale)
    return model


@torch.no_grad()
def probe_opacity_radius_trade(model: MaterialSurfelField, args: argparse.Namespace) -> MaterialSurfelField:
    idx = choose_indices(model, args.sample_fraction, args.seed)
    scale = float(args.opacity_radius_scale)
    alpha = torch.sigmoid(model.raw_alpha[idx])
    alpha_new = (alpha / (scale * scale)).clamp(1e-5, 1.0 - 1e-5)
    model.raw_alpha[idx] = torch.logit(alpha_new)
    model.log_radius[idx] += math.log(scale)
    return model


@torch.no_grad()
def probe_basis_scale_gauge(model: MaterialSurfelField, args: argparse.Namespace) -> MaterialSurfelField:
    if model.L == 0:
        return model
    basis = min(max(0, int(args.basis_index)), model.L - 1)
    scale = float(args.basis_scale_factor)
    model.nr_coeff[:, basis] *= scale
    model.nr_basis[:, basis, :] /= scale
    return model


@torch.no_grad()
def probe_motion_phase_shift(model: MaterialSurfelField, args: argparse.Namespace) -> MaterialSurfelField:
    if model.L == 0 or model.T <= 1:
        return model
    model.nr_coeff.copy_(torch.roll(model.nr_coeff, shifts=int(args.time_shift), dims=0))
    return model


@torch.no_grad()
def probe_opacity_split_clone(model: MaterialSurfelField, args: argparse.Namespace) -> MaterialSurfelField:
    idx = choose_indices(model, args.sample_fraction, args.seed)
    alpha_parent = torch.sigmoid(model.raw_alpha[idx])
    alpha_child = (1.0 - torch.sqrt((1.0 - alpha_parent).clamp_min(1e-6))).clamp(1e-5, 1.0 - 1e-5)
    child_raw_alpha = torch.logit(alpha_child)

    direction = model.x0[idx] / model.x0[idx].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    offset = float(args.split_offset_eps) * direction
    child_plus_x0 = model.x0[idx] + offset
    child_minus_x0 = model.x0[idx] - offset
    x0_new = torch.cat([child_plus_x0, child_minus_x0], dim=0)
    color_new = torch.cat([model.color_logits[idx], model.color_logits[idx]], dim=0)
    raw_alpha_new = torch.cat([child_raw_alpha, child_raw_alpha], dim=0)
    log_radius_new = torch.cat([model.log_radius[idx], model.log_radius[idx]], dim=0)
    basis_new = torch.cat([model.nr_basis[idx], model.nr_basis[idx]], dim=0)

    # The parent is zeroed so the two children replace its opacity footprint.
    model.raw_alpha[idx] = torch.logit(torch.full_like(alpha_parent, 1e-5))
    return append_elements(model, x0_new, color_new, raw_alpha_new, log_radius_new, basis_new)


@torch.no_grad()
def probe_dormant_insert(model: MaterialSurfelField, args: argparse.Namespace) -> MaterialSurfelField:
    idx = choose_indices(model, args.dormant_fraction, args.seed + 17)
    x0 = model.x0[idx].clone()
    x0[:, 2] += float(args.dormant_depth_offset)
    color = model.color_logits[idx].clone()
    raw_alpha = torch.full_like(model.raw_alpha[idx], float(args.dormant_alpha_logit))
    log_radius = model.log_radius[idx].clone()
    basis = model.nr_basis[idx].clone()
    return append_elements(model, x0, color, raw_alpha, log_radius, basis)


PROBES: dict[str, ProbeFn] = {
    "depth_slide": probe_depth_slide,
    "radius_inflate": probe_radius_inflate,
    "opacity_radius_trade": probe_opacity_radius_trade,
    "basis_scale_gauge": probe_basis_scale_gauge,
    "motion_phase_shift": probe_motion_phase_shift,
    "opacity_split_clone": probe_opacity_split_clone,
    "dormant_insert": probe_dormant_insert,
}


@torch.no_grad()
def collect_metrics(
    model: MaterialSurfelField,
    rendered: dict[str, torch.Tensor],
    target: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    cfg: RenderConfig,
    xmap_bins: int,
    xmap_alpha_min: float,
) -> dict[str, float]:
    metrics = {
        **video_metrics(rendered["rgb"], target),
        **alpha_metrics(rendered["alpha"]),
        **model_metrics(model),
        **projection_health_metrics(model, K=K, w2c=w2c, cfg=cfg),
        **motion_health_metrics(model),
        **xmap_health_metrics(
            rendered["xmap"],
            rendered["alpha"],
            canonical_x0=model.x0,
            bins=xmap_bins,
            alpha_min=xmap_alpha_min,
        ),
    }
    if "flow" in rendered:
        metrics.update(flow_health_metrics(rendered["flow"], rendered["alpha"], alpha_min=xmap_alpha_min))
    return metrics


def delta_metrics(
    base_rendered: dict[str, torch.Tensor],
    probe_rendered: dict[str, torch.Tensor],
    base_metrics: dict[str, float],
    probe_metrics: dict[str, float],
) -> dict[str, float]:
    out = {
        "delta_render_l1": float((probe_rendered["rgb"] - base_rendered["rgb"]).abs().mean().detach().cpu()),
        "delta_render_mse": float(((probe_rendered["rgb"] - base_rendered["rgb"]) ** 2).mean().detach().cpu()),
        "delta_target_l1": float(probe_metrics["eval_l1"] - base_metrics["eval_l1"]),
        "delta_alpha_l1": float((probe_rendered["alpha"] - base_rendered["alpha"]).abs().mean().detach().cpu()),
        "delta_depth_l1": float((probe_rendered["depth"] - base_rendered["depth"]).abs().mean().detach().cpu()),
        "delta_xmap_l1": float((probe_rendered["xmap"] - base_rendered["xmap"]).abs().mean().detach().cpu()),
    }
    for key in ("xmap_occ", "projection_coverage_budget", "motion_delta_mean", "motion_coeff_velocity_mean"):
        if key in base_metrics and key in probe_metrics:
            out[f"delta_{key}"] = float(probe_metrics[key] - base_metrics[key])
    return out


def save_probe_strip(
    path: Path,
    target: torch.Tensor,
    base: torch.Tensor,
    probe: torch.Tensor,
    alpha: torch.Tensor,
    max_frames: int = 4,
) -> None:
    T, H, W, _ = target.shape
    count = min(max_frames, T)
    indices = torch.linspace(0, T - 1, count).round().long().tolist()
    rows = []
    for index in indices:
        diff = (probe[index] - base[index]).abs()
        alpha_rgb = alpha[index][..., None].expand(H, W, 3)
        rows.append(torch.cat([target[index], base[index], probe[index], diff, alpha_rgb], dim=1))
    canvas = torch.cat(rows, dim=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_uint8_image(canvas).save(path)
    path.with_name(path.stem + "_columns.txt").write_text(
        "columns: target | base_render | probe_render | abs_probe_minus_base | probe_alpha\n"
    )


def save_rgb_mp4(path: Path, video: torch.Tensor, fps: float = 4.0) -> None:
    import cv2

    frames_u8 = (video.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    _, H, W, _ = frames_u8.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for frame in frames_u8:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def normalize_to_rgb(values: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    data = values.detach()
    if valid is None:
        valid = torch.isfinite(data).all(dim=-1) if data.ndim >= 1 and data.shape[-1] == 3 else torch.isfinite(data)
    if data.shape[-1] == 3:
        flat = data[valid]
        if flat.numel() == 0:
            return torch.zeros_like(data)
        lo = flat.amin(dim=0)
        hi = flat.amax(dim=0)
        return ((data - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    flat = data[valid]
    if flat.numel() == 0:
        return torch.zeros(*data.shape, 3, device=data.device, dtype=data.dtype)
    lo = flat.amin()
    hi = flat.amax()
    norm = ((data - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    return norm[..., None].expand(*data.shape, 3)


def flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    magnitude = flow.norm(dim=-1)
    angle = torch.atan2(flow[..., 1], flow[..., 0])
    hue = (angle + math.pi) / (2.0 * math.pi)
    sat = torch.ones_like(hue)
    val = (magnitude / torch.quantile(magnitude.reshape(-1).detach().cpu(), 0.95).to(flow.device).clamp_min(1e-6)).clamp(0, 1)

    h6 = hue * 6.0
    i = torch.floor(h6).long() % 6
    f = h6 - torch.floor(h6)
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)

    rgb = torch.zeros(*flow.shape[:-1], 3, device=flow.device, dtype=flow.dtype)
    cases = [
        (0, torch.stack([val, t, p], dim=-1)),
        (1, torch.stack([q, val, p], dim=-1)),
        (2, torch.stack([p, val, t], dim=-1)),
        (3, torch.stack([p, q, val], dim=-1)),
        (4, torch.stack([t, p, val], dim=-1)),
        (5, torch.stack([val, p, q], dim=-1)),
    ]
    for case, color in cases:
        rgb = torch.where((i == case)[..., None], color, rgb)
    return rgb.clamp(0, 1)


def save_diagnostic_strips(
    output_dir: Path,
    rendered: dict[str, torch.Tensor],
    alpha_min: float,
    max_frames: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    T = rendered["rgb"].shape[0]
    count = min(max_frames, T)
    indices = torch.linspace(0, T - 1, count).round().long().tolist()

    xmap_rows = []
    for index in indices:
        alpha = rendered["alpha"][index]
        xmap_rgb = normalize_to_rgb(rendered["xmap"][index], valid=alpha > alpha_min)
        alpha_rgb = alpha[..., None].expand_as(xmap_rgb)
        depth_rgb = normalize_to_rgb(rendered["depth"][index], valid=alpha > alpha_min)
        xmap_rows.append(torch.cat([xmap_rgb, depth_rgb, alpha_rgb], dim=1))
    tensor_to_uint8_image(torch.cat(xmap_rows, dim=0)).save(output_dir / "xmap_depth_alpha.png")
    (output_dir / "xmap_depth_alpha_columns.txt").write_text("columns: xmap_rgb | depth | alpha\n")

    if "flow" in rendered:
        flow_rows = []
        flow_T = rendered["flow"].shape[0]
        flow_indices = [min(index, flow_T - 1) for index in indices if flow_T > 0]
        for index in flow_indices:
            flow_rgb = flow_to_rgb(rendered["flow"][index])
            mag_rgb = normalize_to_rgb(rendered["flow"][index].norm(dim=-1))
            flow_rows.append(torch.cat([flow_rgb, mag_rgb], dim=1))
        if flow_rows:
            tensor_to_uint8_image(torch.cat(flow_rows, dim=0)).save(output_dir / "flow.png")
            (output_dir / "flow_columns.txt").write_text("columns: flow_hsv | flow_magnitude\n")


def run_probe(
    probe_name: str,
    base_model: MaterialSurfelField,
    base_rendered: dict[str, torch.Tensor],
    base_metrics: dict[str, float],
    target: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    render_cfg: RenderConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    model = PROBES[probe_name](clone_model(base_model), args)
    rendered = render_sequence(model, K=K, w2c=w2c, cfg=render_cfg, include_flow=args.include_flow)
    metrics = collect_metrics(
        model,
        rendered,
        target,
        K=K,
        w2c=w2c,
        cfg=render_cfg,
        xmap_bins=args.xmap_bins,
        xmap_alpha_min=args.xmap_alpha_min,
    )
    deltas = delta_metrics(base_rendered, rendered, base_metrics, metrics)

    probe_dir = output_dir / probe_name
    save_probe_strip(probe_dir / "preview.png", target, base_rendered["rgb"], rendered["rgb"], rendered["alpha"])
    save_diagnostic_strips(probe_dir, rendered, alpha_min=args.xmap_alpha_min)
    if not args.no_video:
        save_rgb_mp4(probe_dir / "base_render.mp4", base_rendered["rgb"])
        save_rgb_mp4(probe_dir / "probe_render.mp4", rendered["rgb"])
        save_rgb_mp4(probe_dir / "absdiff.mp4", (rendered["rgb"] - base_rendered["rgb"]).abs())
        save_side_by_side_mp4(probe_dir / "target_vs_probe.mp4", target, rendered["rgb"])
    write_json(probe_dir / "probe_metrics.json", {"probe": probe_name, "metrics": metrics, "delta": deltas})
    return {"metrics": metrics, "delta": deltas}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic material-gauge cheat probes on a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe", default="all", choices=["all", *sorted(PROBES)])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-slide-eps", type=float, default=0.02)
    parser.add_argument("--radius-log-scale", type=float, default=0.10)
    parser.add_argument("--opacity-radius-scale", type=float, default=1.20)
    parser.add_argument("--basis-scale-factor", type=float, default=2.0)
    parser.add_argument("--basis-index", type=int, default=0)
    parser.add_argument("--time-shift", type=int, default=1)
    parser.add_argument("--split-offset-eps", type=float, default=0.01)
    parser.add_argument("--dormant-fraction", type=float, default=0.05)
    parser.add_argument("--dormant-depth-offset", type=float, default=0.25)
    parser.add_argument("--dormant-alpha-logit", type=float, default=-4.0)
    parser.add_argument("--xmap-bins", type=int, default=16)
    parser.add_argument("--xmap-alpha-min", type=float, default=0.05)
    parser.add_argument("--include-flow", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = resolve_dynaworld_path(args.checkpoint)
    output_dir = resolve_dynaworld_path(args.output_dir)

    checkpoint = load_checkpoint(checkpoint_path, device)
    cfg = gauge_config(checkpoint["config"])
    target = load_target_video(cfg, device)
    K = checkpoint["K"].to(device)
    w2c = checkpoint["w2c"].to(device)
    render_cfg = render_config_from_checkpoint(checkpoint, cfg)
    base_model = build_model_from_state(checkpoint["model"], device)

    base_rendered = render_sequence(base_model, K=K, w2c=w2c, cfg=render_cfg, include_flow=args.include_flow)
    base_metrics = collect_metrics(
        base_model,
        base_rendered,
        target,
        K=K,
        w2c=w2c,
        cfg=render_cfg,
        xmap_bins=args.xmap_bins,
        xmap_alpha_min=args.xmap_alpha_min,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "base_metrics.json", base_metrics)

    probe_names = sorted(PROBES) if args.probe == "all" else [args.probe]
    results: dict[str, Any] = {"checkpoint": str(checkpoint_path), "base": base_metrics, "probes": {}}
    for probe_name in probe_names:
        print(f"Running probe {probe_name}")
        results["probes"][probe_name] = run_probe(
            probe_name,
            base_model,
            base_rendered,
            base_metrics,
            target,
            K,
            w2c,
            render_cfg,
            args,
            output_dir,
        )

    write_json(output_dir / "probe_summary.json", results)
    print(json.dumps(results["probes"], indent=2, sort_keys=True))
    print(f"Wrote material-gauge probe outputs to {output_dir}")


if __name__ == "__main__":
    main()
