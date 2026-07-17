#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DYNAWORLD = Path(__file__).resolve().parents[2]
LANE_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2"
RESULTS_DIR = LANE_DIR / "results"
TRAIN_EVAL = LANE_DIR / "train_eval_owner_run_tape.py"
COMPARE_STAR = LANE_DIR / "compare_star_uvt_worldfoam_scale.py"
VERIFY_PROMOTION = LANE_DIR / "verify_worldfoam_star_native_cutwalk_promotion.py"
DEFAULT_TAPE_MODE = "owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid"
SUMMARY_SCHEMA_VERSION = "worldfoam_star_native_cutwalk_gate_v2"


def _repo_python() -> Path:
    candidate = DYNAWORLD / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(LANE_DIR)]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _default_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_worldfoam_star_native_cutwalk_%H%M%S")


def _parse_frame_counts(value: str) -> list[int]:
    frame_counts = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            frame_count = int(stripped)
        except ValueError as exc:
            raise ValueError("frame-counts must contain comma-separated integers") from exc
        if frame_count < 1:
            raise ValueError("frame-counts must be positive integers")
        frame_counts.append(frame_count)
    if not frame_counts:
        raise ValueError("frame-counts must contain at least one integer")
    if len(set(frame_counts)) != len(frame_counts):
        raise ValueError("frame-counts must not contain duplicates")
    return frame_counts


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _benchmark_environment_status(payload: dict[str, Any] | None) -> str | None:
    environment = payload.get("benchmark_environment") if payload else None
    return environment.get("status") if isinstance(environment, dict) else None


def _environment_status(environment: dict[str, Any] | None) -> str | None:
    return environment.get("status") if isinstance(environment, dict) else None


def _is_promotable_environment_status(status: Any) -> bool:
    return status in {"ok", "background"}


def _worldfoam_acceptance_failures(payload: dict[str, Any] | None) -> list[str]:
    acceptance = payload.get("acceptance") if isinstance(payload, dict) else None
    if not isinstance(acceptance, dict) or not acceptance:
        return ["WorldFoam artifact acceptance is missing"]
    failed_keys = [key for key, value in acceptance.items() if value is not True]
    if failed_keys:
        return [f"WorldFoam artifact acceptance failed: {','.join(sorted(failed_keys))}"]
    return []


def _blocking_process_summary(environment: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(environment, dict):
        return []
    processes = environment.get("blocking_processes")
    if not isinstance(processes, list):
        return []
    try:
        blocking_cpu_threshold = float(environment.get("blocking_cpu_threshold", 0.0))
    except (TypeError, ValueError):
        blocking_cpu_threshold = 0.0
    rows = []
    for process in processes:
        if not isinstance(process, dict):
            continue
        try:
            pcpu = float(process.get("pcpu", 0.0))
        except (TypeError, ValueError):
            pcpu = 0.0
        high_cpu = pcpu >= blocking_cpu_threshold
        block_reason = process.get("block_reason")
        if block_reason is None and high_cpu:
            block_reason = "high_cpu"
        row = {
            "pid": process.get("pid"),
            "pcpu": process.get("pcpu"),
            "high_cpu": high_cpu,
            "command": process.get("command"),
        }
        if block_reason is not None:
            row["block_reason"] = block_reason
        rows.append(
            row
        )
    rows.sort(key=lambda item: float(item["pcpu"] or 0.0), reverse=True)
    high_cpu_rows = [row for row in rows if row["high_cpu"]]
    persistent_rows = [
        row
        for row in rows
        if not row["high_cpu"] and row.get("block_reason") == "periodic_mps_exporter"
    ]
    return high_cpu_rows + persistent_rows if high_cpu_rows else rows


def _run(cmd: list[str], *, dry_run: bool) -> int:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=DYNAWORLD, env=_env(), check=False).returncode


def _run_json(cmd: list[str], *, dry_run: bool) -> tuple[int, dict[str, Any] | None]:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0, None
    result = subprocess.run(cmd, cwd=DYNAWORLD, env=_env(), check=False, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.stdout:
        print(result.stdout, end="")
    payload = None
    try:
        parsed = json.loads(result.stdout)
        payload = parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        payload = None
    return result.returncode, payload


def _write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_stable_preflight(
    cmd: list[str],
    *,
    sample_count: int,
    interval_s: float,
) -> tuple[int, dict[str, Any] | None, list[dict[str, Any]]]:
    sample_count = max(1, int(sample_count))
    interval_s = max(0.0, float(interval_s))
    samples: list[dict[str, Any]] = []
    last_rc = 2
    last_environment: dict[str, Any] | None = None
    for sample_index in range(1, sample_count + 1):
        last_rc, last_environment = _run_json(cmd, dry_run=False)
        environment_status = _environment_status(last_environment)
        samples.append(
            {
                "sample_index": sample_index,
                "returncode": last_rc,
                "benchmark_environment_status": environment_status,
                "benchmark_environment": last_environment,
                "blocking_processes": _blocking_process_summary(last_environment),
            }
        )
        if last_rc != 0 or not _is_promotable_environment_status(environment_status):
            break
        if sample_index < sample_count and interval_s > 0.0:
            time.sleep(interval_s)
    return last_rc, last_environment, samples


def _worldfoam_preflight_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(_repo_python()),
        str(TRAIN_EVAL),
        "--benchmark-environment-check-only",
        "--wait-for-benchmark-environment-ok-timeout-s",
        str(args.wait_timeout_s),
        "--wait-for-benchmark-environment-ok-poll-s",
        str(args.wait_poll_s),
    ]
    if args.worldfoam_config is not None:
        cmd.extend(["--config", str(args.worldfoam_config)])
    return cmd


def _worldfoam_command(args: argparse.Namespace, out_json: Path) -> list[str]:
    cmd = [
        str(_repo_python()),
        str(TRAIN_EVAL),
        "--frame-counts",
        args.frame_counts,
        "--render-size",
        str(args.render_size),
        "--site-count",
        str(args.site_count),
        "--steps",
        str(args.worldfoam_steps),
        "--warmup-steps",
        str(args.worldfoam_warmup_steps),
        "--optimizer-mode",
        "manual-vjp",
        "--tape-mode",
        DEFAULT_TAPE_MODE,
        "--endpoint-record-source",
        "slow-owner-run",
        "--experimental-selected-only-owner-run-delta-prep",
        "--experimental-native-owner-run-cutwalk-delta",
        "--require-benchmark-environment-ok",
        "--wait-for-benchmark-environment-ok-timeout-s",
        str(args.wait_timeout_s),
        "--wait-for-benchmark-environment-ok-poll-s",
        str(args.wait_poll_s),
        "--post-run-benchmark-environment-settle-s",
        str(args.post_run_benchmark_environment_settle_s),
        "--out-json",
        str(out_json),
    ]
    if args.worldfoam_config is not None:
        cmd.extend(["--config", str(args.worldfoam_config)])
    if args.repeat_loaded_frames:
        cmd.append("--repeat-loaded-frames")
    return cmd


def _star_compare_command(args: argparse.Namespace, worldfoam_json: Path, out_json: Path) -> list[str]:
    cmd = [
        str(_repo_python()),
        str(COMPARE_STAR),
        "--worldfoam-artifact",
        str(worldfoam_json),
        "--frame-counts",
        args.frame_counts,
        "--steps",
        str(args.star_steps),
        "--warmup-steps",
        str(args.star_warmup_steps),
        "--star-target-size",
        str(args.star_target_size),
        "--star-tube-count",
        str(args.star_tube_count),
        "--require-clean-worldfoam-artifact",
        "--require-benchmark-environment-ok",
        "--wait-for-benchmark-environment-ok-timeout-s",
        str(args.wait_timeout_s),
        "--wait-for-benchmark-environment-ok-poll-s",
        str(args.wait_poll_s),
        "--post-run-benchmark-environment-settle-s",
        str(args.post_run_benchmark_environment_settle_s),
        "--out-json",
        str(out_json),
    ]
    if args.star_video_path is not None:
        cmd.extend(["--video-path", str(args.star_video_path)])
    if args.repeat_loaded_frames:
        cmd.append("--star-repeat-loaded-frames")
    return cmd


def _promotion_verifier_command(summary_path: Path) -> list[str]:
    return [
        str(_repo_python()),
        str(VERIFY_PROMOTION),
        str(summary_path),
    ]


def _worldfoam_artifact_path(run_id: str, attempt_index: int, max_attempts: int) -> Path:
    suffix = ".worldfoam.json" if max_attempts <= 1 else f".attempt{attempt_index}.worldfoam.json"
    return RESULTS_DIR / f"{run_id}{suffix}"


def _star_compare_artifact_path(run_id: str, attempt_index: int, max_attempts: int) -> Path:
    suffix = ".star_compare.json" if max_attempts <= 1 else f".star_attempt{attempt_index}.star_compare.json"
    return RESULTS_DIR / f"{run_id}{suffix}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a clean window, run native-cutwalk WorldFoam, then run matched STAR UVT comparison."
    )
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=64)
    parser.add_argument("--site-count", type=int, default=24)
    parser.add_argument("--worldfoam-steps", type=int, default=8)
    parser.add_argument("--worldfoam-warmup-steps", type=int, default=4)
    parser.add_argument("--star-steps", type=int, default=20)
    parser.add_argument("--star-warmup-steps", type=int, default=5)
    parser.add_argument("--star-target-size", type=int, default=64)
    parser.add_argument("--star-tube-count", type=int, default=896)
    parser.add_argument(
        "--worldfoam-config",
        type=Path,
        help=(
            "Pass a non-default train_eval_owner_run_tape config. Use this for a real longer "
            "multicam fixture when one exists; the default checked-in multicam fixture is 16f."
        ),
    )
    parser.add_argument(
        "--star-video-path",
        type=Path,
        help="Pass a non-default STAR source video, for example a real 64f clip for 32f/64f STAR scaling.",
    )
    parser.add_argument("--wait-timeout-s", type=float, default=3600.0)
    parser.add_argument("--wait-poll-s", type=float, default=30.0)
    parser.add_argument(
        "--max-worldfoam-attempts",
        type=int,
        default=1,
        help=(
            "Retry WorldFoam when preflight times out or the artifact ends contended. "
            "STAR runs only after a promotable WorldFoam artifact."
        ),
    )
    parser.add_argument(
        "--max-star-attempts",
        type=int,
        default=1,
        help="Retry STAR comparison when the STAR artifact ends contended after a clean WorldFoam artifact.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only run the benchmark-environment preflight and write a summary; do not launch WorldFoam or STAR.",
    )
    parser.add_argument(
        "--repeat-loaded-frames",
        action="store_true",
        help=(
            "Allow WorldFoam and STAR to repeat a shorter loaded fixture when requested frame counts exceed "
            "the real fixture. This is a synthetic speed-scaling smoke, not a real longer-video quality run."
        ),
    )
    parser.add_argument(
        "--require-real-loaded-frames",
        action="store_true",
        help=(
            "Require promotion verification to reject synthetic repeated-frame scaling. The selected "
            "WorldFoam and STAR artifacts must report loaded_frame_count >= requested frame count and no repeat flags."
        ),
    )
    parser.add_argument(
        "--verify-promotion",
        action="store_true",
        help="After a successful STAR comparison, run the promotion-summary verifier before returning success.",
    )
    parser.add_argument(
        "--preflight-stability-samples",
        type=int,
        default=1,
        help=(
            "Require this many consecutive clean benchmark preflight samples before "
            "launching WorldFoam. Use >1 for clean timing gates."
        ),
    )
    parser.add_argument(
        "--preflight-stability-interval-s",
        type=float,
        default=0.0,
        help="Seconds to wait between consecutive clean preflight samples.",
    )
    parser.add_argument(
        "--post-run-benchmark-environment-settle-s",
        type=float,
        default=2.0,
        help=(
            "Forwarded to WorldFoam and STAR. Lets a transient MTLCompilerService-only "
            "post-run snapshot settle before promotion, while preserving real Python/Torch/MPS blockers."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args(argv)
    if args.require_real_loaded_frames and args.repeat_loaded_frames:
        parser.error("--require-real-loaded-frames cannot be combined with --repeat-loaded-frames")
    if args.require_real_loaded_frames and (args.worldfoam_config is None or args.star_video_path is None):
        parser.error("--require-real-loaded-frames requires --worldfoam-config and --star-video-path")
    try:
        _parse_frame_counts(str(args.frame_counts))
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = str(args.run_id)
    frame_counts = _parse_frame_counts(str(args.frame_counts))
    max_worldfoam_attempts = max(1, int(args.max_worldfoam_attempts))
    max_star_attempts = max(1, int(args.max_star_attempts))
    worldfoam_json = _worldfoam_artifact_path(run_id, 1, max_worldfoam_attempts)
    star_json = _star_compare_artifact_path(run_id, 1, max_star_attempts)
    summary_path = args.summary_json or RESULTS_DIR / f"{run_id}.promotion_summary.json"
    worldfoam_preflight_cmd = _worldfoam_preflight_command(args)
    worldfoam_cmd = _worldfoam_command(args, worldfoam_json)
    star_cmd = _star_compare_command(args, worldfoam_json, star_json)
    summary: dict[str, Any] = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "started",
        "planned_worldfoam_artifact": str(worldfoam_json),
        "worldfoam_artifact": None,
        "worldfoam_promotable_artifact": None,
        "worldfoam_latest_attempt_artifact": None,
        "worldfoam_latest_written_artifact": None,
        "planned_star_compare_artifact": str(star_json),
        "star_compare_artifact": None,
        "star_compare_latest_attempt_artifact": None,
        "star_compare_latest_written_artifact": None,
        "summary_json": str(summary_path),
        "worldfoam_preflight_command": worldfoam_preflight_cmd,
        "worldfoam_command": worldfoam_cmd,
        "planned_star_compare_command": star_cmd,
        "star_compare_command": None,
        "repeat_loaded_frames": bool(args.repeat_loaded_frames),
        "require_real_loaded_frames": bool(args.require_real_loaded_frames),
        "frame_counts": frame_counts,
        "worldfoam_config": str(args.worldfoam_config) if args.worldfoam_config is not None else None,
        "star_video_path": str(args.star_video_path) if args.star_video_path is not None else None,
        "max_worldfoam_attempts": max_worldfoam_attempts,
        "max_star_attempts": max_star_attempts,
        "worldfoam_preflight_stability_samples_required": max(1, int(args.preflight_stability_samples)),
        "worldfoam_preflight_stability_interval_s": max(0.0, float(args.preflight_stability_interval_s)),
        "post_run_benchmark_environment_settle_s": max(
            0.0, float(args.post_run_benchmark_environment_settle_s)
        ),
        "worldfoam_attempts": [],
        "star_compare_attempts": [],
    }
    if args.dry_run:
        summary["status"] = "dry_run"
        _write_summary(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.preflight_only:
        preflight_rc, preflight_environment, preflight_samples = _run_stable_preflight(
            worldfoam_preflight_cmd,
            sample_count=int(args.preflight_stability_samples),
            interval_s=float(args.preflight_stability_interval_s),
        )
        preflight_environment_status = _environment_status(preflight_environment)
        summary.update(
            {
                "status": "preflight_ok"
                if preflight_rc == 0 and _is_promotable_environment_status(preflight_environment_status)
                else "worldfoam_preflight_failed_or_contended",
                "worldfoam_preflight_returncode": preflight_rc,
                "worldfoam_preflight_benchmark_environment_status": preflight_environment_status,
                "worldfoam_preflight_benchmark_environment": preflight_environment,
                "worldfoam_preflight_blocking_processes": _blocking_process_summary(preflight_environment),
                "worldfoam_preflight_samples": preflight_samples,
            }
        )
        _write_summary(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "preflight_ok" else (preflight_rc or 2)

    worldfoam_payload: dict[str, Any] | None = None
    worldfoam_environment: str | None = None
    for attempt_index in range(1, max_worldfoam_attempts + 1):
        worldfoam_json = _worldfoam_artifact_path(run_id, attempt_index, max_worldfoam_attempts)
        worldfoam_preflight_cmd = _worldfoam_preflight_command(args)
        worldfoam_cmd = _worldfoam_command(args, worldfoam_json)
        summary["worldfoam_latest_attempt_artifact"] = str(worldfoam_json)
        summary["worldfoam_preflight_command"] = worldfoam_preflight_cmd
        summary["worldfoam_command"] = worldfoam_cmd
        preflight_rc, preflight_environment, preflight_samples = _run_stable_preflight(
            worldfoam_preflight_cmd,
            sample_count=int(args.preflight_stability_samples),
            interval_s=float(args.preflight_stability_interval_s),
        )
        preflight_environment_status = _environment_status(preflight_environment)
        if preflight_rc != 0 or not _is_promotable_environment_status(preflight_environment_status):
            attempt = {
                "attempt_index": attempt_index,
                "artifact": str(worldfoam_json),
                "artifact_written": False,
                "promotable": False,
                "preflight_returncode": preflight_rc,
                "preflight_benchmark_environment_status": preflight_environment_status,
                "preflight_benchmark_environment": preflight_environment,
                "preflight_samples": preflight_samples,
                "preflight_blocking_processes": _blocking_process_summary(preflight_environment),
                "returncode": None,
                "status": None,
                "benchmark_environment_status": None,
                "acceptance_ok": None,
                "acceptance_failures": [],
            }
            summary["worldfoam_attempts"].append(attempt)
            summary.update(
                {
                    "worldfoam_preflight_returncode": preflight_rc,
                    "worldfoam_preflight_benchmark_environment_status": preflight_environment_status,
                    "worldfoam_preflight_benchmark_environment": preflight_environment,
                    "worldfoam_preflight_blocking_processes": attempt["preflight_blocking_processes"],
                    "worldfoam_preflight_samples": preflight_samples,
                    "worldfoam_returncode": None,
                    "worldfoam_status": None,
                    "worldfoam_benchmark_environment_status": None,
                }
            )
            if attempt_index < max_worldfoam_attempts:
                summary["status"] = "retrying_worldfoam"
                _write_summary(summary_path, summary)
                continue
            summary["status"] = "worldfoam_preflight_failed_or_contended"
            _write_summary(summary_path, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return preflight_rc or 2
        worldfoam_rc = _run(worldfoam_cmd, dry_run=False)
        worldfoam_payload = _load_json(worldfoam_json)
        worldfoam_environment = _benchmark_environment_status(worldfoam_payload)
        artifact_written = worldfoam_payload is not None
        acceptance_failures = _worldfoam_acceptance_failures(worldfoam_payload) if artifact_written else []
        acceptance_ok = artifact_written and not acceptance_failures
        artifact_promotable = (
            worldfoam_rc == 0
            and worldfoam_payload is not None
            and worldfoam_payload.get("status") == "ok"
            and _is_promotable_environment_status(worldfoam_environment)
            and acceptance_ok
        )
        attempt = {
            "attempt_index": attempt_index,
            "artifact": str(worldfoam_json),
            "artifact_written": artifact_written,
            "promotable": artifact_promotable,
            "acceptance_ok": acceptance_ok,
            "acceptance_failures": acceptance_failures,
            "preflight_returncode": preflight_rc,
            "preflight_benchmark_environment_status": preflight_environment_status,
            "preflight_benchmark_environment": preflight_environment,
            "preflight_samples": preflight_samples,
            "preflight_blocking_processes": _blocking_process_summary(preflight_environment),
            "returncode": worldfoam_rc,
            "status": worldfoam_payload.get("status") if worldfoam_payload else None,
            "benchmark_environment_status": worldfoam_environment,
        }
        summary["worldfoam_attempts"].append(attempt)
        summary.update(
            {
                "worldfoam_latest_written_artifact": str(worldfoam_json)
                if artifact_written
                else summary.get("worldfoam_latest_written_artifact"),
                "worldfoam_returncode": worldfoam_rc,
                "worldfoam_status": attempt["status"],
                "worldfoam_benchmark_environment_status": worldfoam_environment,
                "worldfoam_acceptance_ok": acceptance_ok,
                "worldfoam_acceptance_failures": acceptance_failures,
            }
        )
        if artifact_promotable:
            summary["worldfoam_artifact"] = str(worldfoam_json)
            summary["worldfoam_promotable_artifact"] = str(worldfoam_json)
            summary["planned_star_compare_command"] = _star_compare_command(args, worldfoam_json, star_json)
            break
        if attempt_index < max_worldfoam_attempts and (
            (worldfoam_rc == 2 and worldfoam_payload is None)
            or (
                worldfoam_payload is not None
                and not _is_promotable_environment_status(worldfoam_environment)
            )
        ):
            summary["status"] = "retrying_worldfoam"
            _write_summary(summary_path, summary)
            continue
        if worldfoam_payload is not None and (
            worldfoam_payload.get("status") != "ok"
            or not _is_promotable_environment_status(worldfoam_environment)
            or not acceptance_ok
        ):
            summary["status"] = "worldfoam_not_promotable"
            _write_summary(summary_path, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 2
        if worldfoam_rc != 0 or worldfoam_payload is None:
            summary["status"] = (
                "worldfoam_preflight_failed_or_contended"
                if worldfoam_rc == 2 and worldfoam_payload is None
                else "worldfoam_failed"
            )
            _write_summary(summary_path, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return worldfoam_rc or 1

    star_rc = 1
    for attempt_index in range(1, max_star_attempts + 1):
        star_json = _star_compare_artifact_path(run_id, attempt_index, max_star_attempts)
        star_cmd = _star_compare_command(args, worldfoam_json, star_json)
        summary["star_compare_command"] = star_cmd
        summary["star_compare_latest_attempt_artifact"] = str(star_json)
        star_rc = _run(star_cmd, dry_run=False)
        star_payload = _load_json(star_json)
        star_environment = _benchmark_environment_status(star_payload)
        star_artifact_written = star_payload is not None
        star_promotable = (
            star_rc == 0
            and star_payload is not None
            and star_payload.get("status") == "ok"
            and _is_promotable_environment_status(star_environment)
        )
        star_attempt = {
            "attempt_index": attempt_index,
            "artifact": str(star_json),
            "artifact_written": star_artifact_written,
            "promotable": star_promotable,
            "returncode": star_rc,
            "status": star_payload.get("status") if star_payload else None,
            "benchmark_environment_status": star_environment,
            "failures": star_payload.get("failures") if star_payload else None,
        }
        summary["star_compare_attempts"].append(star_attempt)
        summary.update(
            {
                "star_compare_latest_attempt_artifact": str(star_json),
                "star_compare_latest_written_artifact": str(star_json)
                if star_artifact_written
                else summary.get("star_compare_latest_written_artifact"),
                "star_compare_returncode": star_rc,
                "star_compare_status": star_attempt["status"],
                "star_compare_benchmark_environment_status": star_environment,
            }
        )
        if star_promotable:
            summary["star_compare_artifact"] = str(star_json)
            summary["status"] = "ok"
            break
        if (
            attempt_index < max_star_attempts
            and star_payload is not None
            and not _is_promotable_environment_status(star_environment)
        ):
            summary["status"] = "retrying_star_compare"
            _write_summary(summary_path, summary)
            continue
        summary["status"] = "star_compare_failed"
        break
    if args.verify_promotion and summary["status"] == "ok":
        _write_summary(summary_path, summary)
        verifier_cmd = _promotion_verifier_command(summary_path)
        verifier_rc, verifier_payload = _run_json(verifier_cmd, dry_run=False)
        summary.update(
            {
                "promotion_verifier_command": verifier_cmd,
                "promotion_verifier_returncode": verifier_rc,
                "promotion_verifier_status": verifier_payload.get("status") if verifier_payload else None,
                "promotion_verifier_failures": verifier_payload.get("failures") if verifier_payload else None,
            }
        )
        if verifier_rc != 0 or not verifier_payload or verifier_payload.get("status") != "ok":
            summary["status"] = "promotion_verification_failed"
    _write_summary(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ok" else (star_rc or 1)


if __name__ == "__main__":
    raise SystemExit(main())
