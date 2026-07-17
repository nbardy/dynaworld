#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


DYNAWORLD = Path(__file__).resolve().parents[2]
LANE_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2"
RESULTS_DIR = LANE_DIR / "results"
TRAIN_EVAL = LANE_DIR / "train_eval_owner_run_tape.py"
VERIFY_RESULT = LANE_DIR / "verify_worldfoam_next_mps_candidate_result.py"
DEFAULT_READINESS = RESULTS_DIR / "2026-05-20_worldfoam_next_mps_candidate_readiness.json"
DEFAULT_CONFIG = (
    DYNAWORLD
    / "src/train_configs"
    / (
        "local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_"
        "quaternion_height_sv_raytrace_real32_32_smoke.jsonc"
    )
)
DEFAULT_TAPE_MODE = "owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid"


def _repo_python() -> Path:
    candidate = DYNAWORLD / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(LANE_DIR), str(DYNAWORLD / "src/train")]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _default_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_worldfoam_next_mps_%H%M%S")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else DYNAWORLD / path


def _readiness_failures(readiness: dict[str, Any]) -> list[str]:
    failures = []
    if readiness.get("status") != "ok":
        failures.append(f"readiness status is not ok: {readiness.get('status')!r}")
    if readiness.get("ready_for_quiet_mps_quality_speed_run") is not True:
        failures.append("readiness does not mark candidate ready for quiet MPS run")
    if not isinstance(readiness.get("next_mps_candidate"), str) or not readiness["next_mps_candidate"]:
        failures.append("readiness missing next_mps_candidate")
    if readiness.get("quality_claim") is not False:
        failures.append("readiness must keep quality_claim=false before MPS artifact")
    if readiness.get("speed_claim") is not False:
        failures.append("readiness must keep speed_claim=false before MPS artifact")
    if readiness.get("mps_quality_speed_artifact_required") is not True:
        failures.append("readiness must require a clean MPS quality/speed artifact")
    return failures


def _candidate_from_readiness(readiness: dict[str, Any]) -> str:
    failures = _readiness_failures(readiness)
    if failures:
        raise ValueError("; ".join(failures))
    return str(readiness["next_mps_candidate"])


def _preflight_command(args: argparse.Namespace) -> list[str]:
    return [
        str(_repo_python()),
        str(TRAIN_EVAL),
        "--benchmark-environment-check-only",
        "--wait-for-benchmark-environment-ok-timeout-s",
        str(float(args.wait_timeout_s)),
        "--wait-for-benchmark-environment-ok-poll-s",
        str(float(args.wait_poll_s)),
        "--config",
        str(_repo_path(args.config)),
    ]


def _train_eval_command(args: argparse.Namespace, *, candidate: str, out_json: Path) -> list[str]:
    return [
        str(_repo_python()),
        str(TRAIN_EVAL),
        "--config",
        str(_repo_path(args.config)),
        "--frame-counts",
        str(args.frame_counts),
        "--render-size",
        str(int(args.render_size)),
        "--site-count",
        str(int(args.site_count)),
        "--site-initialization",
        candidate,
        "--steps",
        str(int(args.steps)),
        "--warmup-steps",
        str(int(args.warmup_steps)),
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
        str(float(args.wait_timeout_s)),
        "--wait-for-benchmark-environment-ok-poll-s",
        str(float(args.wait_poll_s)),
        "--post-run-benchmark-environment-settle-s",
        str(float(args.post_run_settle_s)),
        "--out-json",
        str(out_json),
    ]


def _result_verifier_command(summary_json: Path) -> list[str]:
    return [str(_repo_python()), str(VERIFY_RESULT), str(summary_json)]


def _extract_json_payload(stdout: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _run_json_command(command: list[str]) -> tuple[int, dict[str, Any] | None, str, str]:
    result = subprocess.run(
        command,
        cwd=DYNAWORLD,
        env=_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode, _extract_json_payload(result.stdout), result.stdout, result.stderr


def _run_result_verifier(command: list[str]) -> tuple[int, dict[str, Any] | None, str, str]:
    return _run_json_command(command)


def _short_text(value: Any, *, limit: int = 1024) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _brief_process(process: Any) -> dict[str, Any]:
    if not isinstance(process, dict):
        return {"raw": _short_text(process)}
    brief: dict[str, Any] = {}
    for key in ("pid", "ppid", "stat", "elapsed", "block_reason", "pcpu", "pmem"):
        if key in process:
            brief[key] = process[key]
    if "command" in process:
        brief["command"] = _short_text(process["command"])
    return brief


def _blocker_kind(process: dict[str, Any]) -> str:
    reason = str(process.get("block_reason") or "")
    command = str(process.get("command") or "").lower()
    if "lean_trade.runners.btc_15m_sft_shadow" in command:
        return "ai_trader_btc15m_sft_shadow"
    if "check_btc15m_sft_runtime_parity.py" in command:
        return "ai_trader_btc15m_sft_runtime_parity"
    if "verify_btc15m_activation_bank_integrity.py" in command:
        return "ai_trader_btc15m_activation_bank_integrity"
    if "build_btc15m_activation_rl_dataset.py" in command:
        return "ai_trader_btc15m_activation_rl"
    if "train_kalshi_btc15m_imitation.py" in command:
        return "ai_trader_btc15m_imitation"
    if "train_kalshi_btc15m_dqn.py" in command:
        return "ai_trader_btc15m_dqn"
    if "train_kalshi_btc15m_sft.py" in command:
        return "ai_trader_btc15m_sft"
    if "font_maker" in command and "train_node_curve_program_flow_v2.py" in command:
        return "font_maker_random_stroke_train"
    if "run_random_stroke_ablation_queue.py" in command:
        return "font_maker_random_stroke_queue"
    if "monitor_standard_glyph_exposure.py" in command:
        return "font_maker_standard_glyph_monitor"
    if (
        reason == "periodic_mps_exporter"
        or "run_btc15m_overnight_shadow_monitor.py" in command
    ):
        return "periodic_mps_exporter"
    if "lean_trade.runners.run_btc_15m_tree_residual_live_quote_shadow_paper" in command:
        return "ai_trader_toto_worker"
    if "btc15m_shadow_overnight" in command and "toto" in command:
        return "ai_trader_toto_worker"
    if "toto" in command:
        return "periodic_mps_exporter"
    if reason.startswith("keyword:torch") or "torch" in command:
        return "torch_worker"
    if reason.startswith("keyword:mps") or " mps" in command or "--device mps" in command:
        return "mps_worker"
    if "mds_stores" in command:
        return "macos_spotlight_indexer"
    if reason.startswith("high_cpu"):
        return "high_cpu_external_job"
    return "other_external_process"


def _screen_session_name(command: Any) -> str | None:
    if not isinstance(command, str) or "screen" not in command.lower():
        return None
    parts = command.split()
    for index, part in enumerate(parts[:-1]):
        if part in {"-dmS", "-S"}:
            return parts[index + 1]
    return None


def _external_blocker_summary(blocking_processes: list[Any]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    screen_session_names: set[str] = set()
    for process in blocking_processes:
        if not isinstance(process, dict):
            kind = "unparsed_process"
            reason = "unparsed"
        else:
            kind = _blocker_kind(process)
            reason = str(process.get("block_reason") or "unknown")
            screen_session = _screen_session_name(process.get("command"))
            if screen_session:
                screen_session_names.add(screen_session)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    manual_next_actions = []
    if kind_counts:
        manual_next_actions.append(
            "rerun only after the benchmark preflight reports a quiet external-process window"
        )
    if kind_counts.get("high_cpu_external_job"):
        manual_next_actions.append("wait for or manually pause high-CPU external training/export jobs")
    if kind_counts.get("ai_trader_btc15m_sft"):
        manual_next_actions.append("wait for ai_trader BTC15M SFT pytest/training workers to finish")
    if kind_counts.get("ai_trader_btc15m_sft_shadow"):
        manual_next_actions.append("wait for ai_trader BTC15M SFT shadow workers to finish")
    if kind_counts.get("ai_trader_btc15m_sft_runtime_parity"):
        manual_next_actions.append("wait for ai_trader BTC15M SFT runtime-parity workers to finish")
    if kind_counts.get("ai_trader_btc15m_activation_bank_integrity"):
        manual_next_actions.append("wait for ai_trader BTC15M activation-bank integrity workers to finish")
    if kind_counts.get("ai_trader_btc15m_imitation"):
        manual_next_actions.append("wait for ai_trader BTC15M imitation pytest/training workers to finish")
    if kind_counts.get("ai_trader_btc15m_dqn"):
        manual_next_actions.append("wait for ai_trader BTC15M DQN pytest/training workers to finish")
    if kind_counts.get("ai_trader_btc15m_activation_rl"):
        manual_next_actions.append("wait for ai_trader BTC15M activation-RL dataset workers to finish")
    if kind_counts.get("font_maker_random_stroke_train"):
        manual_next_actions.append("wait for font_maker random-stroke training to finish or pause it")
    if kind_counts.get("font_maker_random_stroke_queue"):
        manual_next_actions.append("wait for or pause the font_maker random-stroke queue wrapper")
    if kind_counts.get("font_maker_standard_glyph_monitor"):
        manual_next_actions.append("wait for or pause the font_maker standard-glyph monitor")
    if kind_counts.get("ai_trader_toto_worker"):
        manual_next_actions.append("wait for ai_trader/TOTO monitor child workers to finish")
    if kind_counts.get("macos_spotlight_indexer"):
        manual_next_actions.append("wait for macOS Spotlight indexing to cool below the general CPU threshold")
    if kind_counts.get("torch_worker") or kind_counts.get("mps_worker"):
        manual_next_actions.append("wait for or manually pause external torch/MPS workers")
    if kind_counts.get("periodic_mps_exporter"):
        manual_next_actions.append("wait for or manually pause periodic ai_trader/TOTO MPS exporter work")

    return {
        "requires_external_quiet_window": bool(kind_counts),
        "blocking_reason_counts": dict(sorted(reason_counts.items())),
        "blocking_kind_counts": dict(sorted(kind_counts.items())),
        "blocking_screen_session_names": sorted(screen_session_names),
        "manual_next_actions": manual_next_actions,
    }


def _summarize_preflight_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "preflight_benchmark_environment_status": None,
            "preflight_process_sample_limit": None,
            "preflight_blocking_process_count": None,
            "preflight_blocking_process_sample_count": 0,
            "preflight_blocking_process_unlisted_count": 0,
            "preflight_contending_process_count": None,
            "preflight_contending_process_sample_count": 0,
            "preflight_contending_process_unlisted_count": 0,
            "preflight_blocking_reasons": [],
            "preflight_blocking_processes": [],
            "preflight_external_blocker_summary": _external_blocker_summary([]),
        }
    blocking = payload.get("blocking_processes")
    blocking_processes = blocking if isinstance(blocking, list) else []
    contending = payload.get("contending_processes")
    contending_processes = contending if isinstance(contending, list) else []
    blocking_count = payload.get("blocking_process_count")
    if not isinstance(blocking_count, int):
        blocking_count = len(blocking_processes)
    contending_count = payload.get("contending_process_count")
    if not isinstance(contending_count, int):
        contending_count = len(contending_processes)
    process_sample_limit = payload.get("process_sample_limit")
    if not isinstance(process_sample_limit, int):
        process_sample_limit = None
    blocking_sample_count = len(blocking_processes)
    contending_sample_count = len(contending_processes)
    blocking_reasons = sorted(
        {
            str(process["block_reason"])
            for process in blocking_processes
            if isinstance(process, dict) and process.get("block_reason")
        }
    )
    return {
        "preflight_benchmark_environment_status": payload.get("status"),
        "preflight_process_sample_limit": process_sample_limit,
        "preflight_blocking_process_count": blocking_count,
        "preflight_blocking_process_sample_count": blocking_sample_count,
        "preflight_blocking_process_unlisted_count": max(0, blocking_count - blocking_sample_count),
        "preflight_contending_process_count": contending_count,
        "preflight_contending_process_sample_count": contending_sample_count,
        "preflight_contending_process_unlisted_count": max(0, contending_count - contending_sample_count),
        "preflight_blocking_reasons": blocking_reasons,
        "preflight_blocking_processes": [_brief_process(process) for process in blocking_processes],
        "preflight_external_blocker_summary": _external_blocker_summary(blocking_processes),
    }


def _preflight_sample_summary(
    *,
    sample_index: int,
    returncode: int,
    payload: dict[str, Any] | None,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    summary = {
        "sample_index": sample_index,
        "returncode": returncode,
        "stdout_tail": stdout[-1000:],
        "stderr_tail": stderr[-1000:],
    }
    summary.update(_summarize_preflight_payload(payload))
    return summary


def _preflight_attempt_summary(
    *,
    attempt_index: int,
    returncode: int,
    samples: list[dict[str, Any]],
    requested_samples: int,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = {
        "attempt_index": attempt_index,
        "returncode": returncode,
        "samples_completed": len(samples),
        "stability_ok": returncode == 0 and len(samples) == requested_samples,
        "samples": samples,
    }
    summary.update(_summarize_preflight_payload(payload))
    return summary


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
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


def _compact_attempt(attempt: Any) -> dict[str, Any]:
    if not isinstance(attempt, dict):
        return {"raw": _short_text(attempt)}
    blockers = attempt.get("preflight_external_blocker_summary")
    blocking_kind_counts = blockers.get("blocking_kind_counts") if isinstance(blockers, dict) else {}
    blocking_reason_counts = blockers.get("blocking_reason_counts") if isinstance(blockers, dict) else {}
    return {
        "attempt_index": attempt.get("attempt_index"),
        "returncode": attempt.get("returncode"),
        "samples_completed": attempt.get("samples_completed"),
        "stability_ok": attempt.get("stability_ok"),
        "preflight_benchmark_environment_status": attempt.get("preflight_benchmark_environment_status"),
        "preflight_process_sample_limit": attempt.get("preflight_process_sample_limit"),
        "preflight_blocking_process_count": attempt.get("preflight_blocking_process_count"),
        "preflight_blocking_process_sample_count": attempt.get("preflight_blocking_process_sample_count"),
        "preflight_blocking_process_unlisted_count": attempt.get("preflight_blocking_process_unlisted_count"),
        "preflight_contending_process_count": attempt.get("preflight_contending_process_count"),
        "preflight_contending_process_sample_count": attempt.get("preflight_contending_process_sample_count"),
        "preflight_contending_process_unlisted_count": attempt.get("preflight_contending_process_unlisted_count"),
        "blocking_kind_counts": blocking_kind_counts if isinstance(blocking_kind_counts, dict) else {},
        "blocking_reason_counts": blocking_reason_counts if isinstance(blocking_reason_counts, dict) else {},
    }


def _history_entry(payload: dict[str, Any]) -> dict[str, Any]:
    blockers = payload.get("preflight_external_blocker_summary")
    blocking_kind_counts = blockers.get("blocking_kind_counts") if isinstance(blockers, dict) else {}
    blocking_reason_counts = blockers.get("blocking_reason_counts") if isinstance(blockers, dict) else {}
    verifier_payload = payload.get("result_verifier_payload")
    attempts = payload.get("preflight_attempts")
    blocking_processes = payload.get("preflight_blocking_processes")
    return {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "summary_json": payload.get("summary_json"),
        "planned_worldfoam_artifact": payload.get("planned_worldfoam_artifact"),
        "preflight_attempt_count": payload.get("preflight_attempt_count"),
        "preflight_retry_timeout_s": payload.get("preflight_retry_timeout_s"),
        "preflight_returncode": payload.get("preflight_returncode"),
        "preflight_benchmark_environment_status": payload.get("preflight_benchmark_environment_status"),
        "preflight_process_sample_limit": payload.get("preflight_process_sample_limit"),
        "preflight_stability_samples_requested": payload.get("preflight_stability_samples_requested"),
        "preflight_stability_samples_completed": payload.get("preflight_stability_samples_completed"),
        "preflight_stability_ok": payload.get("preflight_stability_ok"),
        "preflight_blocking_process_count": payload.get("preflight_blocking_process_count"),
        "preflight_blocking_process_sample_count": payload.get("preflight_blocking_process_sample_count"),
        "preflight_blocking_process_unlisted_count": payload.get("preflight_blocking_process_unlisted_count"),
        "preflight_contending_process_count": payload.get("preflight_contending_process_count"),
        "preflight_contending_process_sample_count": payload.get("preflight_contending_process_sample_count"),
        "preflight_contending_process_unlisted_count": payload.get("preflight_contending_process_unlisted_count"),
        "preflight_blocking_reasons": payload.get("preflight_blocking_reasons"),
        "preflight_blocking_processes": [_brief_process(process) for process in blocking_processes]
        if isinstance(blocking_processes, list)
        else [],
        "blocking_kind_counts": blocking_kind_counts if isinstance(blocking_kind_counts, dict) else {},
        "blocking_reason_counts": blocking_reason_counts if isinstance(blocking_reason_counts, dict) else {},
        "train_eval_returncode": payload.get("train_eval_returncode"),
        "result_verifier_returncode": payload.get("result_verifier_returncode"),
        "result_verifier_status": verifier_payload.get("status") if isinstance(verifier_payload, dict) else None,
        "preflight_attempts": [_compact_attempt(attempt) for attempt in attempts] if isinstance(attempts, list) else [],
    }


def _append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_history_entry(payload), sort_keys=True) + "\n")


def _write_terminal_summary(path: Path, payload: dict[str, Any]) -> None:
    _write_summary(path, payload)
    history_path = payload.get("history_jsonl")
    if isinstance(history_path, str) and history_path:
        _append_history(Path(history_path), payload)


def _update_preflight_summary(
    summary: dict[str, Any],
    *,
    preflight_rc: int,
    preflight_payload: dict[str, Any] | None,
    preflight_stdout: str,
    preflight_stderr: str,
    preflight_samples: list[dict[str, Any]],
    preflight_attempts: list[dict[str, Any]],
    requested_samples: int,
    retry_deadline: float | None,
) -> None:
    summary.update(
        {
            "preflight_returncode": preflight_rc,
            "preflight_benchmark_environment": preflight_payload,
            "preflight_stdout_tail": preflight_stdout[-4000:],
            "preflight_stderr_tail": preflight_stderr[-4000:],
            "preflight_stability_samples_completed": len(preflight_samples),
            "preflight_stability_ok": preflight_rc == 0 and len(preflight_samples) == requested_samples,
            "preflight_samples": preflight_samples,
            "preflight_attempt_count": len(preflight_attempts),
            "preflight_attempts": preflight_attempts,
            "preflight_retry_remaining_s": max(0.0, retry_deadline - time.monotonic())
            if retry_deadline is not None
            else 0.0,
        }
    )
    summary.update(_summarize_preflight_payload(preflight_payload))


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    readiness_path = _repo_path(args.readiness)
    readiness = _load_json(readiness_path)
    failures = _readiness_failures(readiness)
    candidate = str(readiness.get("next_mps_candidate") or "")
    out_json = _repo_path(args.out_json or RESULTS_DIR / f"{args.run_id}.worldfoam.json")
    summary_json = _repo_path(args.summary_json or RESULTS_DIR / f"{args.run_id}.launch_summary.json")
    history_jsonl = _repo_path(args.history_jsonl) if args.history_jsonl is not None else summary_json.with_suffix(
        ".history.jsonl"
    )
    return {
        "benchmark": "world_foam_next_mps_candidate_launch",
        "status": "readiness_failed" if failures else "planned",
        "failures": failures,
        "run_id": str(args.run_id),
        "readiness_artifact": str(readiness_path),
        "readiness_status": readiness.get("status"),
        "next_mps_candidate": candidate or None,
        "ready_for_quiet_mps_quality_speed_run": readiness.get("ready_for_quiet_mps_quality_speed_run"),
        "quality_claim": readiness.get("quality_claim"),
        "speed_claim": readiness.get("speed_claim"),
        "mps_quality_speed_artifact_required": readiness.get("mps_quality_speed_artifact_required"),
        "planned_worldfoam_artifact": str(out_json),
        "summary_json": str(summary_json),
        "history_jsonl": str(history_jsonl),
        "preflight_command": _preflight_command(args),
        "preflight_stability_samples_requested": int(args.preflight_stability_samples),
        "preflight_stability_interval_s": float(args.preflight_stability_interval_s),
        "preflight_retry_timeout_s": float(args.preflight_retry_timeout_s),
        "preflight_retry_poll_s": float(args.preflight_retry_poll_s),
        "result_verifier_command": _result_verifier_command(summary_json),
        "train_eval_command": _train_eval_command(args, candidate=candidate, out_json=out_json)
        if not failures
        else None,
        "execute": bool(args.execute),
        "verify_result": bool(args.verify_result),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed launcher for the readiness-selected WorldFoam MPS candidate."
    )
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8,16,32")
    parser.add_argument("--render-size", type=int, default=64)
    parser.add_argument("--site-count", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--wait-timeout-s", type=float, default=0.0)
    parser.add_argument("--wait-poll-s", type=float, default=15.0)
    parser.add_argument("--preflight-stability-samples", type=int, default=1)
    parser.add_argument("--preflight-stability-interval-s", type=float, default=5.0)
    parser.add_argument(
        "--preflight-retry-timeout-s",
        type=float,
        default=0.0,
        help=(
            "Retry the whole stability sequence until a clean sequence is observed "
            "or this timeout elapses. The default fails closed after the first "
            "contended sequence."
        ),
    )
    parser.add_argument("--preflight-retry-poll-s", type=float, default=30.0)
    parser.add_argument("--post-run-settle-s", type=float, default=2.0)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--history-jsonl", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-result", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_stability_samples < 1:
        raise ValueError("--preflight-stability-samples must be >= 1")
    if args.preflight_stability_interval_s < 0:
        raise ValueError("--preflight-stability-interval-s must be >= 0")
    if args.preflight_retry_timeout_s < 0:
        raise ValueError("--preflight-retry-timeout-s must be >= 0")
    if args.preflight_retry_poll_s < 0:
        raise ValueError("--preflight-retry-poll-s must be >= 0")
    summary = build_summary(args)
    summary_path = Path(summary["summary_json"])
    if summary["failures"]:
        _write_summary(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1
    if not args.execute:
        summary["status"] = "planned_preflight_only" if args.preflight_only else "planned"
        _write_summary(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    preflight_rc = 0
    preflight_payload = None
    preflight_stdout = ""
    preflight_stderr = ""
    preflight_samples: list[dict[str, Any]] = []
    preflight_attempts: list[dict[str, Any]] = []
    retry_timeout_s = float(args.preflight_retry_timeout_s)
    retry_deadline = time.monotonic() + retry_timeout_s if retry_timeout_s > 0 else None
    attempt_index = 0
    while True:
        attempt_index += 1
        preflight_samples = []
        for sample_index in range(1, int(args.preflight_stability_samples) + 1):
            preflight_rc, preflight_payload, preflight_stdout, preflight_stderr = _run_json_command(
                summary["preflight_command"]
            )
            preflight_samples.append(
                _preflight_sample_summary(
                    sample_index=sample_index,
                    returncode=preflight_rc,
                    payload=preflight_payload,
                    stdout=preflight_stdout,
                    stderr=preflight_stderr,
                )
            )
            if preflight_rc != 0:
                break
            if sample_index < args.preflight_stability_samples and args.preflight_stability_interval_s > 0:
                time.sleep(float(args.preflight_stability_interval_s))
        preflight_attempts.append(
            _preflight_attempt_summary(
                attempt_index=attempt_index,
                returncode=preflight_rc,
                samples=preflight_samples,
                requested_samples=int(args.preflight_stability_samples),
                payload=preflight_payload,
            )
        )
        requested_samples = int(args.preflight_stability_samples)
        if preflight_rc == 0 and len(preflight_samples) == requested_samples:
            break
        if retry_deadline is None or time.monotonic() >= retry_deadline:
            break
        _update_preflight_summary(
            summary,
            preflight_rc=preflight_rc,
            preflight_payload=preflight_payload,
            preflight_stdout=preflight_stdout,
            preflight_stderr=preflight_stderr,
            preflight_samples=preflight_samples,
            preflight_attempts=preflight_attempts,
            requested_samples=requested_samples,
            retry_deadline=retry_deadline,
        )
        summary["status"] = "preflight_retry_waiting"
        _write_terminal_summary(summary_path, summary)
        sleep_s = min(float(args.preflight_retry_poll_s), max(0.0, retry_deadline - time.monotonic()))
        if sleep_s > 0:
            time.sleep(sleep_s)
    _update_preflight_summary(
        summary,
        preflight_rc=preflight_rc,
        preflight_payload=preflight_payload,
        preflight_stdout=preflight_stdout,
        preflight_stderr=preflight_stderr,
        preflight_samples=preflight_samples,
        preflight_attempts=preflight_attempts,
        requested_samples=int(args.preflight_stability_samples),
        retry_deadline=retry_deadline,
    )
    if preflight_rc != 0:
        summary["status"] = "preflight_contended"
        _write_terminal_summary(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return preflight_rc
    if args.preflight_only:
        summary["status"] = "preflight_ok"
        _write_terminal_summary(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    train_cmd = summary["train_eval_command"]
    train_rc = subprocess.run(train_cmd, cwd=DYNAWORLD, env=_env(), check=False).returncode
    summary["train_eval_returncode"] = train_rc
    summary["status"] = "train_eval_ok" if train_rc == 0 else "train_eval_failed"
    if train_rc == 0 and args.verify_result:
        _write_summary(summary_path, summary)
        verifier_rc, verifier_payload, verifier_stdout, verifier_stderr = _run_result_verifier(
            summary["result_verifier_command"]
        )
        summary.update(
            {
                "result_verifier_returncode": verifier_rc,
                "result_verifier_payload": verifier_payload,
                "result_verifier_stdout_tail": verifier_stdout[-4000:],
                "result_verifier_stderr_tail": verifier_stderr[-4000:],
            }
        )
        if verifier_rc != 0:
            summary["status"] = "result_verification_failed"
        _write_terminal_summary(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return verifier_rc
    _write_terminal_summary(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return train_rc


if __name__ == "__main__":
    raise SystemExit(main())
