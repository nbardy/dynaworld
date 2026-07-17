#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

from diagnose_worldfoam_mps_blockers import DEFAULT_RECENT_SECONDS, diagnose_summary
import report_worldfoam_fork_shader_goal_state as goal_report
import run_worldfoam_next_mps_candidate as launcher


DEFAULT_GOAL_STATE_JSON = (
    goal_report.RESULTS_DIR / "2026-05-21_worldfoam_fork_shader_goal_state.json"
)
PROMOTABLE_BENCHMARK_ENVIRONMENT_STATUSES = frozenset({"ok", "background"})


def _short_text(value: Any, *, limit: int = 2048) -> str:
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


def _current_benchmark_environment_probe(
    *,
    config: Path = launcher.DEFAULT_CONFIG,
    wait_timeout_s: float = 0.0,
    wait_poll_s: float = 15.0,
) -> dict[str, Any]:
    args = SimpleNamespace(
        config=config,
        wait_timeout_s=float(wait_timeout_s),
        wait_poll_s=float(wait_poll_s),
    )
    command = launcher._preflight_command(args)
    returncode, payload, stdout, stderr = launcher._run_json_command(command)
    if not isinstance(payload, dict):
        probe = {
            "available": False,
            "command": command,
            "returncode": returncode,
            "status": "unreadable",
            "blocks_promotion": True,
            "stdout": _short_text(stdout),
            "stderr": _short_text(stderr),
        }
        return goal_report._current_benchmark_environment_report(probe) or probe
    status = payload.get("status")
    probe = {
        "available": True,
        "command": command,
        "returncode": returncode,
        "status": status,
        "blocks_promotion": returncode != 0 or status not in PROMOTABLE_BENCHMARK_ENVIRONMENT_STATUSES,
        "blocking_process_count": payload.get("blocking_process_count"),
        "contending_process_count": payload.get("contending_process_count"),
        "background_process_count": payload.get("background_process_count"),
        "process_sample_limit": payload.get("process_sample_limit"),
        "blocking_processes": payload.get("blocking_processes") if isinstance(payload.get("blocking_processes"), list) else [],
        "contending_processes": payload.get("contending_processes")
        if isinstance(payload.get("contending_processes"), list)
        else [],
        "background_processes": payload.get("background_processes")
        if isinstance(payload.get("background_processes"), list)
        else [],
    }
    return goal_report._current_benchmark_environment_report(probe) or probe


def refresh(
    *,
    source_json: Path = goal_report.DEFAULT_SOURCE_JSON,
    import_json: Path = goal_report.DEFAULT_IMPORT_JSON,
    smoke_bundle_json: Path = goal_report.DEFAULT_SMOKE_BUNDLE_JSON,
    next_mps_summary_json: Path | None = None,
    blocker_diagnosis_json: Path = goal_report.DEFAULT_BLOCKER_DIAGNOSIS_JSON,
    out_json: Path = DEFAULT_GOAL_STATE_JSON,
    recent_seconds: float = DEFAULT_RECENT_SECONDS,
    max_blocker_diagnosis_age_s: float | None = goal_report.DEFAULT_MAX_BLOCKER_DIAGNOSIS_AGE_S,
    probe_current_benchmark_environment: bool = True,
    current_benchmark_environment_config: Path = launcher.DEFAULT_CONFIG,
    current_benchmark_environment_wait_timeout_s: float = 0.0,
    current_benchmark_environment_wait_poll_s: float = 15.0,
) -> dict[str, Any]:
    source_json = source_json.resolve(strict=False)
    import_json = import_json.resolve(strict=False)
    smoke_bundle_json = smoke_bundle_json.resolve(strict=False)
    blocker_diagnosis_json = blocker_diagnosis_json.resolve(strict=False)
    out_json = out_json.resolve(strict=False)
    if next_mps_summary_json is None:
        next_mps_summary_json = goal_report.default_next_mps_summary_json()
    else:
        next_mps_summary_json = next_mps_summary_json.resolve(strict=False)

    diagnosis = diagnose_summary(next_mps_summary_json, recent_seconds=recent_seconds)
    _write_json_atomic(blocker_diagnosis_json, diagnosis)
    current_environment_probe = (
        _current_benchmark_environment_probe(
            config=current_benchmark_environment_config,
            wait_timeout_s=current_benchmark_environment_wait_timeout_s,
            wait_poll_s=current_benchmark_environment_wait_poll_s,
        )
        if probe_current_benchmark_environment
        else None
    )

    report = goal_report.audit(
        source_json=source_json,
        import_json=import_json,
        smoke_bundle_json=smoke_bundle_json,
        next_mps_summary_json=next_mps_summary_json,
        blocker_diagnosis_json=blocker_diagnosis_json,
        current_benchmark_environment_probe=current_environment_probe,
        max_blocker_diagnosis_age_s=max_blocker_diagnosis_age_s,
    )
    _write_json_atomic(out_json, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh WorldFoam blocker diagnosis, then regenerate the fork-shader goal report."
    )
    parser.add_argument("--source-json", type=Path, default=goal_report.DEFAULT_SOURCE_JSON)
    parser.add_argument("--import-json", type=Path, default=goal_report.DEFAULT_IMPORT_JSON)
    parser.add_argument("--smoke-bundle-json", type=Path, default=goal_report.DEFAULT_SMOKE_BUNDLE_JSON)
    parser.add_argument("--next-mps-summary-json", type=Path)
    parser.add_argument("--blocker-diagnosis-json", type=Path, default=goal_report.DEFAULT_BLOCKER_DIAGNOSIS_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_GOAL_STATE_JSON)
    parser.add_argument("--recent-seconds", type=float, default=DEFAULT_RECENT_SECONDS)
    parser.add_argument(
        "--skip-current-benchmark-environment-probe",
        action="store_true",
        help="Do not run the current train_eval benchmark-environment preflight while refreshing the report.",
    )
    parser.add_argument(
        "--current-benchmark-environment-config",
        type=Path,
        default=launcher.DEFAULT_CONFIG,
    )
    parser.add_argument("--current-benchmark-environment-wait-timeout-s", type=float, default=0.0)
    parser.add_argument("--current-benchmark-environment-wait-poll-s", type=float, default=15.0)
    parser.add_argument(
        "--max-blocker-diagnosis-age-s",
        type=float,
        default=goal_report.DEFAULT_MAX_BLOCKER_DIAGNOSIS_AGE_S,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = refresh(
        source_json=args.source_json,
        import_json=args.import_json,
        smoke_bundle_json=args.smoke_bundle_json,
        next_mps_summary_json=args.next_mps_summary_json,
        blocker_diagnosis_json=args.blocker_diagnosis_json,
        out_json=args.out_json,
        recent_seconds=float(args.recent_seconds),
        max_blocker_diagnosis_age_s=args.max_blocker_diagnosis_age_s,
        probe_current_benchmark_environment=not bool(args.skip_current_benchmark_environment_probe),
        current_benchmark_environment_config=args.current_benchmark_environment_config,
        current_benchmark_environment_wait_timeout_s=float(args.current_benchmark_environment_wait_timeout_s),
        current_benchmark_environment_wait_poll_s=float(args.current_benchmark_environment_wait_poll_s),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] != "failed_prerequisite" else 1


if __name__ == "__main__":
    raise SystemExit(main())
