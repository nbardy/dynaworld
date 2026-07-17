#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _newest_waiter_summary(results_dir: Path = RESULTS_DIR) -> Path:
    summaries = sorted(
        results_dir.glob("*clean_mps_wait*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not summaries:
        raise FileNotFoundError(f"no clean-gate waiter summary found in {results_dir}")
    return summaries[0]


def _sample_processes(payload: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    sample = payload.get("current_benchmark_environment_blocking_process_sample")
    if not isinstance(sample, list) or limit <= 0:
        return []
    out = []
    for process in sample[:limit]:
        if not isinstance(process, dict):
            continue
        out.append(
            {
                key: process.get(key)
                for key in (
                    "pid",
                    "ppid",
                    "pcpu",
                    "pmem",
                    "elapsed",
                    "block_reason",
                    "screen_session_name",
                    "command",
                )
            }
        )
    return out


def _summary_age_s(path: Path, *, now_ns: int | None = None) -> float:
    now = time.time_ns() if now_ns is None else int(now_ns)
    return max(0.0, (now - path.stat().st_mtime_ns) / 1_000_000_000.0)


def _is_stale_for_poll(payload: dict[str, Any], *, age_s: float) -> bool | None:
    poll_s = payload.get("wait_ready_poll_s")
    if not isinstance(poll_s, (int, float)) or isinstance(poll_s, bool) or poll_s <= 0:
        return None
    return age_s > float(poll_s) * 2.0


def _wait_for_newer_summary(
    *,
    results_dir: Path,
    initial_mtime_ns: int,
    timeout_s: float,
    poll_s: float,
) -> Path:
    deadline_s = time.monotonic() + max(0.0, float(timeout_s))
    sleep_s = max(0.05, float(poll_s))
    path = _newest_waiter_summary(results_dir)
    while path.stat().st_mtime_ns <= initial_mtime_ns and time.monotonic() < deadline_s:
        time.sleep(min(sleep_s, max(0.0, deadline_s - time.monotonic())))
        path = _newest_waiter_summary(results_dir)
    return path


def read_status(
    *,
    include_sample: int = 0,
    wait_refresh_timeout_s: float = 0.0,
    wait_refresh_poll_s: float = 5.0,
) -> dict[str, Any]:
    path = _newest_waiter_summary()
    if wait_refresh_timeout_s > 0:
        path = _wait_for_newer_summary(
            results_dir=path.parent,
            initial_mtime_ns=path.stat().st_mtime_ns,
            timeout_s=wait_refresh_timeout_s,
            poll_s=wait_refresh_poll_s,
        )
    payload = _load_json(path)
    age_s = _summary_age_s(path)
    status = {
        "summary_json": str(path),
        "mtime_ns": path.stat().st_mtime_ns,
        "summary_age_s": age_s,
        "summary_stale_for_poll": _is_stale_for_poll(payload, age_s=age_s),
        "status": payload.get("status"),
        "ready_refresh_count": payload.get("ready_refresh_count"),
        "ready_to_run_now": payload.get("ready_to_run_now"),
        "wait_elapsed_s": payload.get("wait_elapsed_s"),
        "wait_remaining_s": payload.get("wait_remaining_s"),
        "launch_returncode": payload.get("launch_returncode"),
        "launch_summary_json": payload.get("launch_summary_json"),
        "post_launch_goal_status": payload.get("post_launch_goal_status"),
        "post_launch_objective_complete": payload.get("post_launch_objective_complete"),
        "blocking_conditions": payload.get("blocking_conditions"),
        "blockers": payload.get("current_benchmark_environment_blocking_kind_counts"),
        "reason_counts": payload.get("current_benchmark_environment_blocking_reason_counts"),
        "live_remaining": payload.get("live_max_estimated_remaining_s_by_category"),
        "manual_next_actions": payload.get("current_benchmark_environment_manual_next_actions"),
    }
    if include_sample:
        status["process_sample"] = _sample_processes(payload, limit=include_sample)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read the latest clean-gate waiter summary without passing MPS-named paths on argv."
    )
    parser.add_argument("--sample", type=int, default=0, help="Include up to this many blocking process rows.")
    parser.add_argument(
        "--wait-refresh-timeout-s",
        type=float,
        default=0.0,
        help="Wait up to this many seconds for the newest waiter summary to be rewritten before reading it.",
    )
    parser.add_argument(
        "--wait-refresh-poll-s",
        type=float,
        default=5.0,
        help="Polling interval used with --wait-refresh-timeout-s.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            read_status(
                include_sample=max(0, int(args.sample)),
                wait_refresh_timeout_s=max(0.0, float(args.wait_refresh_timeout_s)),
                wait_refresh_poll_s=max(0.05, float(args.wait_refresh_poll_s)),
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
