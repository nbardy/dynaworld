#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import refresh_worldfoam_fork_shader_goal_state as refresh_mod
import report_worldfoam_fork_shader_goal_state as goal_report
from diagnose_worldfoam_mps_blockers import DEFAULT_RECENT_SECONDS


DEFAULT_SUMMARY_JSON = goal_report.RESULTS_DIR / "2026-05-21_worldfoam_clean_mps_ready_gate.json"


def _short_text(value: Any, *, limit: int = 4000) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(encoded)
    try:
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _plan_from_report(report: dict[str, Any]) -> dict[str, Any]:
    plan = report.get("clean_mps_rerun_plan")
    return plan if isinstance(plan, dict) else {}


def _command_from_plan(plan: dict[str, Any]) -> list[str]:
    command = plan.get("command")
    if not isinstance(command, list):
        return []
    return [str(item) for item in command]


def _verification_contract_failures(payload: dict[str, Any]) -> list[str]:
    failures = []
    command = payload.get("command")
    if not isinstance(command, list) or "--verify-result" not in command:
        failures.append("clean MPS command must include --verify-result")
    if payload.get("embedded_result_verification") is not True:
        failures.append("clean MPS plan must mark embedded_result_verification=true")
    if payload.get("acceptance_verifier_required_status") != "ok":
        failures.append("clean MPS verifier required status must be ok")
    template = payload.get("acceptance_verifier_command_template")
    if not isinstance(template, list):
        failures.append("clean MPS verifier command template is missing")
    else:
        joined = " ".join(str(item) for item in template)
        if "verify_worldfoam_next_mps_candidate_result.py" not in joined:
            failures.append("clean MPS verifier command template must call result verifier")
        if "<launch_summary_json>" not in template:
            failures.append("clean MPS verifier command template must include <launch_summary_json>")
    return failures


def _ready_gate_payload(
    *,
    report: dict[str, Any],
    execute: bool,
    goal_state_json: Path,
) -> dict[str, Any]:
    plan = _plan_from_report(report)
    ready = plan.get("ready_to_run_now") is True
    status = "ready_dry_run" if ready and not execute else "ready_to_launch" if ready else "not_ready"
    current_actions = plan.get("current_benchmark_environment_manual_next_actions")
    current_sample = plan.get("current_benchmark_environment_blocking_process_sample")
    return {
        "status": status,
        "execute": execute,
        "goal_state_json": str(goal_state_json),
        "goal_status": report.get("status"),
        "objective_complete": report.get("objective_complete"),
        "ready_to_run_now": plan.get("ready_to_run_now"),
        "blocking_conditions": plan.get("blocking_conditions") if isinstance(plan.get("blocking_conditions"), list) else [],
        "wait_reason": plan.get("wait_reason"),
        "run_after_estimated_done_at": plan.get("run_after_estimated_done_at"),
        "run_after_estimated_done_at_scope": plan.get("run_after_estimated_done_at_scope"),
        "run_after_estimated_done_at_requires_reprobe": plan.get("run_after_estimated_done_at_requires_reprobe"),
        "live_max_estimated_remaining_s_by_category": plan.get("live_max_estimated_remaining_s_by_category")
        if isinstance(plan.get("live_max_estimated_remaining_s_by_category"), dict)
        else {},
        "current_benchmark_environment_status": plan.get("current_benchmark_environment_status"),
        "current_benchmark_environment_blocking_kind_counts": plan.get(
            "current_benchmark_environment_blocking_kind_counts"
        )
        if isinstance(plan.get("current_benchmark_environment_blocking_kind_counts"), dict)
        else {},
        "current_benchmark_environment_blocking_reason_counts": plan.get(
            "current_benchmark_environment_blocking_reason_counts"
        )
        if isinstance(plan.get("current_benchmark_environment_blocking_reason_counts"), dict)
        else {},
        "current_benchmark_environment_blocking_screen_session_names": plan.get(
            "current_benchmark_environment_blocking_screen_session_names"
        )
        if isinstance(plan.get("current_benchmark_environment_blocking_screen_session_names"), list)
        else [],
        "current_benchmark_environment_blocking_process_count": plan.get(
            "current_benchmark_environment_blocking_process_count"
        ),
        "current_benchmark_environment_manual_next_actions": (
            [str(action) for action in current_actions] if isinstance(current_actions, list) else []
        ),
        "current_benchmark_environment_blocking_process_sample": (
            current_sample if isinstance(current_sample, list) else []
        ),
        "live_blocking_category_counts": plan.get("live_blocking_category_counts")
        if isinstance(plan.get("live_blocking_category_counts"), dict)
        else {},
        "live_blocking_screen_session_names": plan.get("live_blocking_screen_session_names")
        if isinstance(plan.get("live_blocking_screen_session_names"), list)
        else [],
        "live_recent_output_category_counts": plan.get("live_recent_output_category_counts")
        if isinstance(plan.get("live_recent_output_category_counts"), dict)
        else {},
        "command": _command_from_plan(plan),
        "acceptance_gate": plan.get("acceptance_gate"),
        "embedded_result_verification": plan.get("embedded_result_verification"),
        "acceptance_verifier_required_status": plan.get("acceptance_verifier_required_status"),
        "acceptance_verifier_command_template": plan.get("acceptance_verifier_command_template")
        if isinstance(plan.get("acceptance_verifier_command_template"), list)
        else [],
    }


def _stdout_payload(payload: dict[str, Any], *, print_payload: str) -> dict[str, Any]:
    if print_payload == "full":
        return payload
    if print_payload != "summary":
        raise ValueError(f"unknown print_payload: {print_payload}")
    keep_keys = (
        "status",
        "execute",
        "goal_state_json",
        "goal_status",
        "objective_complete",
        "ready_to_run_now",
        "blocking_conditions",
        "wait_reason",
        "run_after_estimated_done_at",
        "run_after_estimated_done_at_scope",
        "run_after_estimated_done_at_requires_reprobe",
        "live_max_estimated_remaining_s_by_category",
        "current_benchmark_environment_status",
        "current_benchmark_environment_blocking_kind_counts",
        "current_benchmark_environment_blocking_reason_counts",
        "current_benchmark_environment_blocking_screen_session_names",
        "current_benchmark_environment_blocking_process_count",
        "current_benchmark_environment_manual_next_actions",
        "live_blocking_category_counts",
        "live_blocking_screen_session_names",
        "live_recent_output_category_counts",
        "ready_refresh_count",
        "wait_ready_timeout_s",
        "wait_ready_poll_s",
        "wait_elapsed_s",
        "wait_remaining_s",
        "launch_returncode",
        "post_launch_goal_status",
        "post_launch_objective_complete",
        "acceptance_gate",
        "embedded_result_verification",
        "acceptance_verifier_required_status",
        "acceptance_verifier_command_template",
    )
    return {key: payload[key] for key in keep_keys if key in payload}


def _refresh_report(
    *,
    goal_state_json: Path,
    recent_seconds: float,
    source_json: Path,
    import_json: Path,
    smoke_bundle_json: Path,
    next_mps_summary_json: Path | None,
    blocker_diagnosis_json: Path,
    probe_current_benchmark_environment: bool,
    current_benchmark_environment_config: Path,
    current_benchmark_environment_wait_timeout_s: float,
    current_benchmark_environment_wait_poll_s: float,
    max_blocker_diagnosis_age_s: float | None,
) -> dict[str, Any]:
    return refresh_mod.refresh(
        source_json=source_json,
        import_json=import_json,
        smoke_bundle_json=smoke_bundle_json,
        next_mps_summary_json=next_mps_summary_json,
        blocker_diagnosis_json=blocker_diagnosis_json,
        out_json=goal_state_json,
        recent_seconds=recent_seconds,
        max_blocker_diagnosis_age_s=max_blocker_diagnosis_age_s,
        probe_current_benchmark_environment=probe_current_benchmark_environment,
        current_benchmark_environment_config=current_benchmark_environment_config,
        current_benchmark_environment_wait_timeout_s=current_benchmark_environment_wait_timeout_s,
        current_benchmark_environment_wait_poll_s=current_benchmark_environment_wait_poll_s,
    )


def run_gate_when_ready(
    *,
    execute: bool = False,
    summary_json: Path = DEFAULT_SUMMARY_JSON,
    goal_state_json: Path = refresh_mod.DEFAULT_GOAL_STATE_JSON,
    recent_seconds: float = DEFAULT_RECENT_SECONDS,
    source_json: Path = goal_report.DEFAULT_SOURCE_JSON,
    import_json: Path = goal_report.DEFAULT_IMPORT_JSON,
    smoke_bundle_json: Path = goal_report.DEFAULT_SMOKE_BUNDLE_JSON,
    next_mps_summary_json: Path | None = None,
    blocker_diagnosis_json: Path = goal_report.DEFAULT_BLOCKER_DIAGNOSIS_JSON,
    probe_current_benchmark_environment: bool = True,
    current_benchmark_environment_config: Path = refresh_mod.launcher.DEFAULT_CONFIG,
    current_benchmark_environment_wait_timeout_s: float = 0.0,
    current_benchmark_environment_wait_poll_s: float = 15.0,
    max_blocker_diagnosis_age_s: float | None = goal_report.DEFAULT_MAX_BLOCKER_DIAGNOSIS_AGE_S,
    refresh_after_launch: bool = True,
    wait_ready_timeout_s: float = 0.0,
    wait_ready_poll_s: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    if wait_ready_timeout_s < 0:
        raise ValueError("wait_ready_timeout_s must be >= 0")
    if wait_ready_poll_s < 0:
        raise ValueError("wait_ready_poll_s must be >= 0")
    start_s = time.monotonic()
    deadline_s = start_s + wait_ready_timeout_s if wait_ready_timeout_s > 0 else None
    refresh_count = 0
    while True:
        refresh_count += 1
        report = _refresh_report(
            goal_state_json=goal_state_json,
            recent_seconds=recent_seconds,
            source_json=source_json,
            import_json=import_json,
            smoke_bundle_json=smoke_bundle_json,
            next_mps_summary_json=next_mps_summary_json,
            blocker_diagnosis_json=blocker_diagnosis_json,
            probe_current_benchmark_environment=probe_current_benchmark_environment,
            current_benchmark_environment_config=current_benchmark_environment_config,
            current_benchmark_environment_wait_timeout_s=current_benchmark_environment_wait_timeout_s,
            current_benchmark_environment_wait_poll_s=current_benchmark_environment_wait_poll_s,
            max_blocker_diagnosis_age_s=max_blocker_diagnosis_age_s,
        )
        payload = _ready_gate_payload(report=report, execute=execute, goal_state_json=goal_state_json)
        elapsed_s = max(0.0, time.monotonic() - start_s)
        payload.update(
            {
                "ready_refresh_count": refresh_count,
                "wait_ready_timeout_s": float(wait_ready_timeout_s),
                "wait_ready_poll_s": float(wait_ready_poll_s),
                "wait_elapsed_s": elapsed_s,
            }
        )
        if payload["ready_to_run_now"] is True:
            break
        if deadline_s is None:
            _write_json_atomic(summary_json, payload)
            return 2, payload
        remaining_s = deadline_s - time.monotonic()
        if remaining_s <= 0:
            payload["status"] = "not_ready_timeout"
            payload["wait_remaining_s"] = 0.0
            _write_json_atomic(summary_json, payload)
            return 2, payload
        payload["status"] = "waiting_for_ready"
        payload["wait_remaining_s"] = remaining_s
        _write_json_atomic(summary_json, payload)
        time.sleep(min(wait_ready_poll_s, remaining_s))
    command = payload["command"]
    if not command:
        payload["status"] = "ready_but_missing_command"
        _write_json_atomic(summary_json, payload)
        return 1, payload
    verification_failures = _verification_contract_failures(payload)
    if verification_failures:
        payload["status"] = "ready_but_unverified_command"
        payload["verification_contract_failures"] = verification_failures
        _write_json_atomic(summary_json, payload)
        return 1, payload
    if not execute:
        _write_json_atomic(summary_json, payload)
        return 0, payload

    result = subprocess.run(
        command,
        cwd=goal_report.ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    payload.update(
        {
            "status": "launch_ok" if result.returncode == 0 else "launch_failed",
            "launch_returncode": result.returncode,
            "launch_stdout_tail": _short_text(result.stdout),
            "launch_stderr_tail": _short_text(result.stderr),
        }
    )
    if refresh_after_launch:
        post_report = _refresh_report(
            goal_state_json=goal_state_json,
            recent_seconds=recent_seconds,
            source_json=source_json,
            import_json=import_json,
            smoke_bundle_json=smoke_bundle_json,
            next_mps_summary_json=next_mps_summary_json,
            blocker_diagnosis_json=blocker_diagnosis_json,
            probe_current_benchmark_environment=probe_current_benchmark_environment,
            current_benchmark_environment_config=current_benchmark_environment_config,
            current_benchmark_environment_wait_timeout_s=current_benchmark_environment_wait_timeout_s,
            current_benchmark_environment_wait_poll_s=current_benchmark_environment_wait_poll_s,
            max_blocker_diagnosis_age_s=max_blocker_diagnosis_age_s,
        )
        payload["post_launch_goal_status"] = post_report.get("status")
        payload["post_launch_objective_complete"] = post_report.get("objective_complete")
    _write_json_atomic(summary_json, payload)
    return result.returncode, payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the WorldFoam goal report and run the clean MPS gate only if ready_to_run_now is true."
    )
    parser.add_argument("--execute", action="store_true", help="Run the embedded clean MPS command if ready.")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--goal-state-json", type=Path, default=refresh_mod.DEFAULT_GOAL_STATE_JSON)
    parser.add_argument("--recent-seconds", type=float, default=DEFAULT_RECENT_SECONDS)
    parser.add_argument("--source-json", type=Path, default=goal_report.DEFAULT_SOURCE_JSON)
    parser.add_argument("--import-json", type=Path, default=goal_report.DEFAULT_IMPORT_JSON)
    parser.add_argument("--smoke-bundle-json", type=Path, default=goal_report.DEFAULT_SMOKE_BUNDLE_JSON)
    parser.add_argument("--next-mps-summary-json", type=Path)
    parser.add_argument("--blocker-diagnosis-json", type=Path, default=goal_report.DEFAULT_BLOCKER_DIAGNOSIS_JSON)
    parser.add_argument("--skip-current-benchmark-environment-probe", action="store_true")
    parser.add_argument("--current-benchmark-environment-config", type=Path, default=refresh_mod.launcher.DEFAULT_CONFIG)
    parser.add_argument("--current-benchmark-environment-wait-timeout-s", type=float, default=0.0)
    parser.add_argument("--current-benchmark-environment-wait-poll-s", type=float, default=15.0)
    parser.add_argument("--max-blocker-diagnosis-age-s", type=float, default=goal_report.DEFAULT_MAX_BLOCKER_DIAGNOSIS_AGE_S)
    parser.add_argument(
        "--skip-refresh-after-launch",
        action="store_true",
        help="Do not refresh the goal report again after a launched gate exits.",
    )
    parser.add_argument(
        "--wait-ready-timeout-s",
        type=float,
        default=0.0,
        help="Poll refreshed goal state until ready_to_run_now is true or this timeout expires.",
    )
    parser.add_argument(
        "--wait-ready-poll-s",
        type=float,
        default=30.0,
        help="Sleep interval between ready-state refreshes when --wait-ready-timeout-s is positive.",
    )
    parser.add_argument(
        "--print-payload",
        choices=("summary", "full"),
        default="summary",
        help="Print a compact status by default; the full payload is always written to --summary-json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    returncode, payload = run_gate_when_ready(
        execute=bool(args.execute),
        summary_json=args.summary_json,
        goal_state_json=args.goal_state_json,
        recent_seconds=float(args.recent_seconds),
        source_json=args.source_json,
        import_json=args.import_json,
        smoke_bundle_json=args.smoke_bundle_json,
        next_mps_summary_json=args.next_mps_summary_json,
        blocker_diagnosis_json=args.blocker_diagnosis_json,
        probe_current_benchmark_environment=not bool(args.skip_current_benchmark_environment_probe),
        current_benchmark_environment_config=args.current_benchmark_environment_config,
        current_benchmark_environment_wait_timeout_s=float(args.current_benchmark_environment_wait_timeout_s),
        current_benchmark_environment_wait_poll_s=float(args.current_benchmark_environment_wait_poll_s),
        max_blocker_diagnosis_age_s=args.max_blocker_diagnosis_age_s,
        refresh_after_launch=not bool(args.skip_refresh_after_launch),
        wait_ready_timeout_s=float(args.wait_ready_timeout_s),
        wait_ready_poll_s=float(args.wait_ready_poll_s),
    )
    print(json.dumps(_stdout_payload(payload, print_payload=args.print_payload), indent=2, sort_keys=True))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
