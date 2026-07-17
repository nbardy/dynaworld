from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

import gate4_affine_slab_tape as gate4
import train_eval_owner_run_tape as bench


def _prepare(
    *,
    use_sorted: bool,
    sites: tuple[Any, ...],
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    site_rgba_cpu: torch.Tensor,
    tape_mode: str,
) -> dict[str, Any]:
    return bench._prepare_owner_run_tapes(
        sites=sites,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
        site_rgba=site_rgba_cpu,
        tape_mode=tape_mode,
        endpoint_record_source="gate4-affine",
        gate4_time_slabs=1,
        gate4_residual_depth_padding=0.001,
        experimental_native_sorted_delta=bool(use_sorted),
    )


def _tensor_equal_report(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {"all_equal": True, "mismatches": []}
    shared_keys = sorted(set(lhs["selected_device"]).intersection(rhs["selected_device"]))
    for key in shared_keys:
        left = lhs["selected_device"][key]
        right = rhs["selected_device"][key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            continue
        same_shape = tuple(left.shape) == tuple(right.shape)
        same_dtype = left.dtype == right.dtype
        same = same_shape and same_dtype and bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
        if not same:
            report["all_equal"] = False
            report["mismatches"].append(
                {
                    "key": key,
                    "left_shape": list(left.shape),
                    "right_shape": list(right.shape),
                    "left_dtype": str(left.dtype),
                    "right_dtype": str(right.dtype),
                }
            )
    return report


def _clone_selected_device_tensors(tape: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(tape)
    cloned["selected_device"] = {
        key: value.clone().contiguous() if isinstance(value, torch.Tensor) else value
        for key, value in tape["selected_device"].items()
    }
    torch.mps.synchronize()
    return cloned


def _time_vjp(
    *,
    tape: dict[str, Any],
    site_rgba: torch.Tensor,
    target_rgb_track: torch.Tensor,
    op_config: bench.RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    timings: list[float] = []
    loss_values: list[float] = []
    for index in range(warmup + repeat):
        torch.mps.synchronize()
        start = time.perf_counter()
        loss, grad = bench._delta_replace_coeff16_fused_mse_loss_vjp(
            tape_device=tape["selected_device"],
            site_rgba=site_rgba,
            target_rgb_track=target_rgb_track,
            op_config=op_config,
            track_count=track_count,
            frame_count=frame_count,
        )
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        if index >= warmup:
            timings.append(elapsed)
            loss_values.append(float(loss.detach().cpu().item()))
            if float(grad.detach().abs().sum().cpu().item()) <= 0.0:
                raise RuntimeError("VJP produced zero gradient")
    values = torch.tensor(timings, dtype=torch.float64)
    return {
        "count": int(values.numel()),
        "mean_ms": float(values.mean().item() * 1000.0),
        "median_ms": float(values.median().item() * 1000.0),
        "min_ms": float(values.min().item() * 1000.0),
        "max_ms": float(values.max().item() * 1000.0),
        "loss_first": loss_values[0],
        "loss_last": loss_values[-1],
    }


def _device_inputs(
    *,
    site_rgba_cpu: torch.Tensor,
    targets: torch.Tensor,
    frame_count: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    sync_before_device: bool,
    collect_before_device: bool,
) -> tuple[torch.Tensor, torch.Tensor, bench.RealRayReplayConfig]:
    if sync_before_device:
        torch.mps.synchronize()
    if collect_before_device:
        gc.collect()
    device = torch.device("mps")
    site_rgba = site_rgba_cpu.to(device=device).contiguous()
    train_targets = targets.to(device=device)
    train_view_count = int(train_targets.shape[0] // frame_count)
    target_rgb_track = bench._track_major_rgb_from_image(
        train_targets,
        view_count=train_view_count,
        frame_count=frame_count,
        height=int(train_targets.shape[2]),
        width=int(train_targets.shape[3]),
    )
    op_config = bench.RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    return site_rgba, target_rgb_track, op_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--render-size", type=int, default=64)
    parser.add_argument("--site-count", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=("compare", "cut", "sorted"),
        default="compare",
        help="'compare' keeps both tapes resident; 'cut'/'sorted' time one tape in a clean process.",
    )
    parser.add_argument("--collect-before-device", action="store_true")
    parser.add_argument("--sync-before-device", action="store_true")
    parser.add_argument("--clone-device-tensors", action="store_true")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    frame_count = int(args.frame_count)
    near = 0.1
    far = 6.0
    density = 10.0
    invalid_epsilon = 1.0e-6
    transmittance_threshold = 1.0e-4
    tape_mode = "endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse"

    cfg = bench._load_config(bench.DEFAULT_CONFIG, max_frames=frame_count, render_size=int(args.render_size))
    data = bench.load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    targets, rays, frame_indices, _repeated = bench._fit_loaded_frame_count(
        split_name="train",
        targets=targets,
        rays=rays,
        frame_indices=frame_indices,
        loaded_frame_count=int(data["frame_count"]),
        requested_frame_count=frame_count,
        allow_repeat_loaded_frames=False,
    )
    rays = bench.apply_synthetic_ray_motion(
        rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        motion=bench.SyntheticRayMotion(
            origin_velocity=(0.08, 0.0, 0.02),
            direction_velocity=(0.02, 0.0, 0.0),
        ),
    )
    sites = bench.initialize_sites_from_train_samples(
        targets=targets,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        site_count=int(args.site_count),
        near=near,
        far=far,
        density=density,
    )
    site_rgba_cpu = torch.tensor([site.rgba for site in sites], dtype=torch.float32)

    if args.mode in {"cut", "sorted"}:
        use_sorted = args.mode == "sorted"
        tape = _prepare(
            use_sorted=use_sorted,
            sites=sites,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=frame_count,
            near=near,
            far=far,
            invalid_epsilon=invalid_epsilon,
            transmittance_threshold=transmittance_threshold,
            site_rgba_cpu=site_rgba_cpu,
            tape_mode=tape_mode,
        )
        site_rgba, target_rgb_track, op_config = _device_inputs(
            site_rgba_cpu=site_rgba_cpu,
            targets=targets,
            frame_count=frame_count,
            near=near,
            far=far,
            invalid_epsilon=invalid_epsilon,
            transmittance_threshold=transmittance_threshold,
            sync_before_device=bool(args.sync_before_device),
            collect_before_device=bool(args.collect_before_device),
        )
        if args.clone_device_tensors:
            tape = _clone_selected_device_tensors(tape)
        stats = _time_vjp(
            tape=tape,
            site_rgba=site_rgba,
            target_rgb_track=target_rgb_track,
            op_config=op_config,
            track_count=int(tape["track_count"]),
            frame_count=frame_count,
            warmup=int(args.warmup),
            repeat=int(args.repeat),
        )
        payload = {
            "mode": args.mode,
            "frame_count": frame_count,
            "render_size": int(args.render_size),
            "site_count": int(args.site_count),
            "default_sorted_enabled_after_probe": bool(gate4.GATE4_ENABLE_EXPERIMENTAL_NATIVE_SORTED_DELTA),
            "collect_before_device": bool(args.collect_before_device),
            "sync_before_device": bool(args.sync_before_device),
            "clone_device_tensors": bool(args.clone_device_tensors),
            "prepare_timings": tape["prepare_timings"],
            "timing": stats,
            "selected_segments": int(tape["selected_segments"]),
            "selected_storage_bytes": int(tape["selected_storage_bytes"]),
            "endpoint_record_delta_replace_change_events": int(tape["endpoint_record_delta_replace_change_events"]),
            "endpoint_record_delta_replace_changed_records": int(
                tape["endpoint_record_delta_replace_changed_records"]
            ),
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.out_json is not None:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(text + "\n", encoding="utf-8")
        print(text)
        return

    cut_tape = _prepare(
        use_sorted=False,
        sites=sites,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
        site_rgba_cpu=site_rgba_cpu,
        tape_mode=tape_mode,
    )
    sorted_tape = _prepare(
        use_sorted=True,
        sites=sites,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
        site_rgba_cpu=site_rgba_cpu,
        tape_mode=tape_mode,
    )
    equality = _tensor_equal_report(cut_tape, sorted_tape)
    site_rgba, target_rgb_track, op_config = _device_inputs(
        site_rgba_cpu=site_rgba_cpu,
        targets=targets,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
        sync_before_device=bool(args.sync_before_device),
        collect_before_device=bool(args.collect_before_device),
    )
    if args.clone_device_tensors:
        cut_tape = _clone_selected_device_tensors(cut_tape)
        sorted_tape = _clone_selected_device_tensors(sorted_tape)

    cut_stats = _time_vjp(
        tape=cut_tape,
        site_rgba=site_rgba,
        target_rgb_track=target_rgb_track,
        op_config=op_config,
        track_count=int(cut_tape["track_count"]),
        frame_count=frame_count,
        warmup=int(args.warmup),
        repeat=int(args.repeat),
    )
    sorted_stats = _time_vjp(
        tape=sorted_tape,
        site_rgba=site_rgba,
        target_rgb_track=target_rgb_track,
        op_config=op_config,
        track_count=int(sorted_tape["track_count"]),
        frame_count=frame_count,
        warmup=int(args.warmup),
        repeat=int(args.repeat),
    )
    cut_stats_second = _time_vjp(
        tape=cut_tape,
        site_rgba=site_rgba,
        target_rgb_track=target_rgb_track,
        op_config=op_config,
        track_count=int(cut_tape["track_count"]),
        frame_count=frame_count,
        warmup=0,
        repeat=int(args.repeat),
    )

    payload = {
        "frame_count": frame_count,
        "render_size": int(args.render_size),
        "site_count": int(args.site_count),
        "default_sorted_enabled_after_probe": bool(gate4.GATE4_ENABLE_EXPERIMENTAL_NATIVE_SORTED_DELTA),
        "collect_before_device": bool(args.collect_before_device),
        "sync_before_device": bool(args.sync_before_device),
        "clone_device_tensors": bool(args.clone_device_tensors),
        "tensor_equality": equality,
        "prepare_timings": {
            "cut": cut_tape["prepare_timings"],
            "sorted": sorted_tape["prepare_timings"],
        },
        "timing": {
            "cut_first": cut_stats,
            "sorted_second": sorted_stats,
            "cut_third": cut_stats_second,
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
