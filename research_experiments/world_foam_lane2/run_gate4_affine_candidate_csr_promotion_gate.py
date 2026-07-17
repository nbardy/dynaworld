#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from train_eval_owner_run_tape import (
    DEFAULT_CONFIG,
    GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE,
    RESULTS_DIR,
    SyntheticRayMotion,
    _benchmark_environment_blocks_promotion,
    _capture_benchmark_environment,
    run_train_eval,
)
from verify_gate4_affine_candidate_csr_train_eval import verify as verify_candidate_csr


def _default_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_gate4_affine_candidate_csr_promotion_%H%M%S")


def _parse_int_list(value: str) -> tuple[int, ...]:
    out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not out:
        raise ValueError("expected at least one frame count")
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compact_process(process: dict[str, Any]) -> dict[str, Any]:
    return {
        "pid": process.get("pid"),
        "pcpu": process.get("pcpu"),
        "pmem": process.get("pmem"),
        "command": process.get("command"),
    }


def _compact_environment(environment: dict[str, Any]) -> dict[str, Any]:
    blocking = environment.get("blocking_processes")
    contending = environment.get("contending_processes")
    background = environment.get("background_processes")
    compact_blocking = [_compact_process(item) for item in blocking[:5]] if isinstance(blocking, list) else []
    compact_contending = [_compact_process(item) for item in contending[:5]] if isinstance(contending, list) else []
    blocking_count = environment.get("blocking_process_count")
    if not isinstance(blocking_count, int):
        blocking_count = len(blocking) if isinstance(blocking, list) else 0
    contending_count = environment.get("contending_process_count")
    if not isinstance(contending_count, int):
        contending_count = len(contending) if isinstance(contending, list) else 0
    background_count = environment.get("background_process_count")
    if not isinstance(background_count, int):
        background_count = len(background) if isinstance(background, list) else 0
    return {
        "status": environment.get("status", "unknown"),
        "pid": environment.get("pid"),
        "blocking_cpu_threshold": environment.get("blocking_cpu_threshold"),
        "blocking_process_count": blocking_count,
        "contending_process_count": contending_count,
        "background_process_count": background_count,
        "blocking_processes": compact_blocking,
        "contending_processes": compact_contending,
    }


def _wait_for_benchmark_environment(args: argparse.Namespace) -> dict[str, Any]:
    snapshots: list[dict[str, Any]] = []
    stable_checks = 0
    deadline = time.monotonic() + float(args.wait_timeout_s)
    while True:
        environment = _capture_benchmark_environment()
        compact_environment = _compact_environment(environment)
        snapshots.append(compact_environment)
        if not _benchmark_environment_blocks_promotion(environment):
            stable_checks += 1
        else:
            stable_checks = 0
        if stable_checks >= int(args.stable_preflight_checks):
            return {
                "status": "ok",
                "stable_checks": stable_checks,
                "snapshots": snapshots,
                "final_environment": compact_environment,
            }
        if not args.wait_for_benchmark_environment_ok:
            return {
                "status": "blocked",
                "stable_checks": stable_checks,
                "snapshots": snapshots,
                "final_environment": compact_environment,
            }
        if time.monotonic() >= deadline:
            return {
                "status": "timeout",
                "stable_checks": stable_checks,
                "snapshots": snapshots,
                "final_environment": compact_environment,
            }
        print(
            "[run_gate4_affine_candidate_csr_promotion_gate] "
            f"benchmark environment {environment.get('status')}; waiting {args.wait_interval_s}s",
            flush=True,
        )
        time.sleep(float(args.wait_interval_s))


def _verify_args(args: argparse.Namespace, *, artifact: Path, verify_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        artifact=artifact,
        frame_counts=args.frame_counts,
        render_size=int(args.render_size),
        site_count=int(args.site_count),
        min_train_psnr=float(args.min_train_psnr),
        min_heldout_psnr=float(args.min_heldout_psnr),
        max_total_scale=float(args.max_total_scale),
        max_backward_scale=float(args.max_backward_scale),
        max_total_median_scale=float(args.max_total_median_scale),
        max_backward_median_scale=float(args.max_backward_median_scale),
        max_storage_scale=float(args.max_storage_scale),
        max_noncoeff_storage_scale=float(args.max_noncoeff_storage_scale),
        max_candidate_scale=float(args.max_candidate_scale),
        max_candidates_per_row=int(args.max_candidates_per_row),
        max_row_mean_to_median=float(args.max_row_mean_to_median),
        max_row_max_to_median=float(args.max_row_max_to_median),
        tape_mode=GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE,
        allow_contended=False,
        out_json=verify_path,
    )


def _run_attempt(args: argparse.Namespace, *, attempt: int) -> dict[str, Any]:
    results_dir = Path(args.results_dir)
    artifact = results_dir / f"{args.run_id}.attempt{attempt}.json"
    verify_path = results_dir / f"{args.run_id}.attempt{attempt}.verify.json"
    payload = run_train_eval(
        config_path=Path(args.config),
        frame_counts=_parse_int_list(str(args.frame_counts)),
        render_size=int(args.render_size),
        site_count=int(args.site_count),
        near=float(args.near),
        far=float(args.far),
        density=float(args.density),
        invalid_epsilon=float(args.invalid_epsilon),
        transmittance_threshold=float(args.transmittance_threshold),
        synthetic_motion=SyntheticRayMotion(
            origin_velocity=(
                float(args.origin_velocity_x),
                float(args.origin_velocity_y),
                float(args.origin_velocity_z),
            ),
            direction_velocity=(
                float(args.direction_velocity_x),
                float(args.direction_velocity_y),
                float(args.direction_velocity_z),
            ),
        ),
        steps=int(args.steps),
        warmup_steps=int(args.warmup_steps),
        lr=float(args.lr),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        adam_eps=float(args.adam_eps),
        optimizer_mode="manual-vjp",
        segment_tape_vjp_mode="direct_atomic_grad_only",
        tape_mode=GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE,
        allow_repeat_loaded_frames=False,
        endpoint_record_source="gate4-affine",
        gate4_time_slabs=int(args.gate4_time_slabs),
        gate4_residual_depth_padding=float(args.gate4_residual_depth_padding),
        defer_heldout_device=bool(args.defer_heldout_device),
    )
    _write_json(artifact, payload)
    verify_result = verify_candidate_csr(_verify_args(args, artifact=artifact, verify_path=verify_path))
    if not verify_path.exists():
        _write_json(verify_path, verify_result)
    return {
        "attempt": int(attempt),
        "artifact": str(artifact),
        "verify_artifact": str(verify_path),
        "artifact_status": payload.get("status"),
        "verify_status": verify_result.get("status"),
        "verify_failures": verify_result.get("failures", []),
        "verify_contamination": verify_result.get("contamination", []),
        "total_step_scale": verify_result.get("total_step_scale"),
        "backward_scale": verify_result.get("backward_scale"),
        "resident_noncoeff_storage_scale": verify_result.get("resident_noncoeff_storage_scale"),
        "candidate_count_scale": verify_result.get("candidate_count_scale"),
    }


def run_promotion(args: argparse.Namespace) -> dict[str, Any]:
    results_dir = Path(args.results_dir)
    summary_path = Path(args.out_summary) if args.out_summary is not None else results_dir / f"{args.run_id}.promotion_summary.json"
    attempts: list[dict[str, Any]] = []
    preflight_results: list[dict[str, Any]] = []
    status = "verify_failed"
    for attempt in range(1, int(args.max_promotion_attempts) + 1):
        preflight = _wait_for_benchmark_environment(args)
        preflight_results.append(preflight)
        if preflight["status"] != "ok":
            status = "preflight_blocked"
            break
        print(
            "[run_gate4_affine_candidate_csr_promotion_gate] "
            f"attempt {attempt}/{args.max_promotion_attempts}",
            flush=True,
        )
        attempt_result = _run_attempt(args, attempt=attempt)
        attempts.append(attempt_result)
        if attempt_result["verify_status"] == "ok":
            status = "promoted"
            break
        contamination = attempt_result.get("verify_contamination")
        if not contamination:
            status = "verify_failed"
            break
        status = "contaminated_retry_exhausted"
    summary = {
        "run_id": args.run_id,
        "status": status,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "preflight_results": preflight_results,
        "summary_path": str(summary_path),
        "tape_mode": GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE,
        "endpoint_record_source": "gate4-affine",
        "frame_counts": _parse_int_list(str(args.frame_counts)),
        "render_size": int(args.render_size),
        "site_count": int(args.site_count),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
    }
    _write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and verify the Gate4 affine candidate CSR promotion gate.")
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-summary", type=Path)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=16)
    parser.add_argument("--site-count", type=int, default=24)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--density", type=float, default=10.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--gate4-time-slabs", type=int, default=1)
    parser.add_argument("--gate4-residual-depth-padding", type=float, default=0.001)
    parser.add_argument("--no-defer-heldout-device", action="store_false", dest="defer_heldout_device")
    parser.set_defaults(defer_heldout_device=True)
    parser.add_argument("--max-promotion-attempts", type=int, default=3)
    parser.add_argument("--wait-for-benchmark-environment-ok", action="store_true", default=True)
    parser.add_argument("--no-wait-for-benchmark-environment-ok", action="store_false", dest="wait_for_benchmark_environment_ok")
    parser.add_argument("--wait-timeout-s", type=float, default=900.0)
    parser.add_argument("--wait-interval-s", type=float, default=30.0)
    parser.add_argument("--stable-preflight-checks", type=int, default=2)
    parser.add_argument("--min-train-psnr", type=float, default=8.0)
    parser.add_argument("--min-heldout-psnr", type=float, default=8.0)
    parser.add_argument("--max-total-scale", type=float, default=2.0)
    parser.add_argument("--max-backward-scale", type=float, default=2.0)
    parser.add_argument("--max-total-median-scale", type=float, default=2.0)
    parser.add_argument("--max-backward-median-scale", type=float, default=2.0)
    parser.add_argument("--max-storage-scale", type=float, default=1.10)
    parser.add_argument("--max-noncoeff-storage-scale", type=float, default=1.10)
    parser.add_argument("--max-candidate-scale", type=float, default=1.10)
    parser.add_argument("--max-candidates-per-row", type=int, default=256)
    parser.add_argument("--max-row-mean-to-median", type=float, default=2.0)
    parser.add_argument("--max-row-max-to-median", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    summary = run_promotion(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "promoted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
