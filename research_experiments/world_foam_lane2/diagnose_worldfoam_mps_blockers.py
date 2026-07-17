#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research_experiments" / "world_foam_lane2" / "results"
DEFAULT_RECENT_SECONDS = 15 * 60
RECENT_FILE_LIMIT = 12
ACTIVE_CPU_THRESHOLD = 5.0
KEYWORD_BLOCKING_CPU_THRESHOLD = 5.0
GENERAL_BLOCKING_CPU_THRESHOLD = 75.0


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


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


def _short_text(value: Any, *, limit: int = 1024) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _summary_payload(path: Path) -> tuple[dict[str, Any], Path]:
    payload = _load_json(path)
    next_mps = (
        payload.get("artifacts", {}).get("next_mps_quality_speed")
        if isinstance(payload.get("artifacts"), dict)
        else None
    )
    next_path = next_mps.get("path") if isinstance(next_mps, dict) else None
    if isinstance(next_path, str) and next_path:
        resolved = Path(next_path)
        if not resolved.is_absolute():
            resolved = path.parent / resolved
        return _load_json(resolved), resolved
    return payload, path


def _processes_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    processes = summary.get("preflight_blocking_processes")
    if isinstance(processes, list):
        return [process for process in processes if isinstance(process, dict)]
    environment = summary.get("preflight_benchmark_environment")
    if isinstance(environment, dict):
        processes = environment.get("blocking_processes")
        if isinstance(processes, list):
            return [process for process in processes if isinstance(process, dict)]
    return []


def _summary_int(summary: dict[str, Any], key: str, *, default: int) -> int:
    value = summary.get(key)
    return value if isinstance(value, int) and value >= 0 else default


def _parse_ps_line(line: str) -> dict[str, Any] | None:
    parts = line.split(None, 6)
    if len(parts) < 7:
        return None
    try:
        return {
            "pid": int(parts[0]),
            "ppid": int(parts[1]),
            "stat": parts[2],
            "elapsed": parts[3],
            "pcpu": float(parts[4]),
            "pmem": float(parts[5]),
            "command": parts[6],
        }
    except ValueError:
        return None


def _capture_live_processes(pids: list[int]) -> dict[int, dict[str, Any]]:
    if not pids:
        return {}
    command = [
        "ps",
        "-ww",
        "-p",
        ",".join(str(pid) for pid in sorted(set(pids))),
        "-o",
        "pid=,ppid=,stat=,etime=,pcpu=,pmem=,command=",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=3.0)
    except (OSError, subprocess.SubprocessError):
        return {}
    live: dict[int, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        row = _parse_ps_line(line)
        if row is not None:
            live[int(row["pid"])] = row
    return live


def _cpu_blocking_threshold(block_reason: str) -> float | None:
    if block_reason == "high_cpu_general":
        return GENERAL_BLOCKING_CPU_THRESHOLD
    if block_reason == "high_cpu":
        return KEYWORD_BLOCKING_CPU_THRESHOLD
    return None


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _extract_option(command: str, option: str) -> str | None:
    parts = _split_command(command)
    for index, part in enumerate(parts[:-1]):
        if part == option:
            return parts[index + 1]
    prefix = option + "="
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _parse_ps_elapsed_seconds(elapsed: Any) -> float | None:
    if not isinstance(elapsed, str) or not elapsed:
        return None
    day_split = elapsed.split("-", 1)
    if len(day_split) == 2:
        try:
            days = int(day_split[0])
        except ValueError:
            return None
        rest = day_split[1]
    else:
        days = 0
        rest = elapsed
    pieces = rest.split(":")
    try:
        ints = [int(piece) for piece in pieces]
    except ValueError:
        return None
    if len(ints) == 2:
        hours = 0
        minutes, seconds = ints
    elif len(ints) == 3:
        hours, minutes, seconds = ints
    else:
        return None
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


def _declared_duration_hours(command: str) -> float | None:
    raw = _extract_option(command, "--duration-hours")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0.0 else None


def _runtime_estimate(command: str, elapsed: Any, *, now_s: float) -> dict[str, Any]:
    duration_hours = _declared_duration_hours(command)
    elapsed_s = _parse_ps_elapsed_seconds(elapsed)
    if duration_hours is None and elapsed_s is None:
        return {}
    estimate: dict[str, Any] = {
        "declared_duration_hours": duration_hours,
        "elapsed_s": elapsed_s,
    }
    if duration_hours is not None and elapsed_s is not None:
        duration_s = duration_hours * 3600.0
        remaining_s = max(0.0, duration_s - elapsed_s)
        estimate["declared_duration_s"] = duration_s
        estimate["estimated_remaining_s"] = remaining_s
        estimate["estimated_done_at"] = datetime.fromtimestamp(now_s + remaining_s).astimezone().isoformat(
            timespec="seconds"
        )
    return estimate


def _extract_cd_cwd(command: str) -> Path | None:
    match = re.search(r"(?:^|\s)cd\s+([^;&]+?)\s*&&", command)
    if not match:
        return None
    raw = match.group(1).strip().strip("'\"")
    return Path(raw) if raw else None


def _resolve_command_path(command: str, raw_path: str | None, *, cwd: Path | None = None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    direct_cwd = _extract_cd_cwd(command)
    resolved_cwd = direct_cwd if direct_cwd is not None else cwd
    return (resolved_cwd / path) if resolved_cwd is not None else path


def _resolve_output_dir(command: str, cwd: Path | None, known_cwds: list[Path]) -> Path | None:
    output_dir = _resolve_command_path(command, _extract_option(command, "--output-dir"), cwd=cwd)
    if output_dir is None or output_dir.is_absolute() or output_dir.exists():
        return output_dir
    for known_cwd in known_cwds:
        candidate = known_cwd / output_dir
        if candidate.exists():
            return candidate
    return output_dir


def _process_pid(process: dict[str, Any]) -> int | None:
    pid = process.get("pid")
    return pid if isinstance(pid, int) else None


def _process_ppid(process: dict[str, Any]) -> int | None:
    ppid = process.get("ppid")
    return ppid if isinstance(ppid, int) else None


def _direct_cwd_by_pid(processes: list[dict[str, Any]]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for process in processes:
        pid = _process_pid(process)
        if pid is None:
            continue
        cwd = _extract_cd_cwd(str(process.get("command") or ""))
        if cwd is not None:
            out[pid] = cwd
    return out


def _parent_by_pid(processes: list[dict[str, Any]]) -> dict[int, int]:
    out: dict[int, int] = {}
    for process in processes:
        pid = _process_pid(process)
        ppid = _process_ppid(process)
        if pid is not None and ppid is not None:
            out[pid] = ppid
    return out


def _inherited_cwd(
    process: dict[str, Any],
    *,
    direct_cwds: dict[int, Path],
    parents: dict[int, int],
) -> tuple[Path | None, str | None]:
    pid = _process_pid(process)
    seen: set[int] = set()
    while pid is not None and pid not in seen:
        seen.add(pid)
        cwd = direct_cwds.get(pid)
        if cwd is not None:
            return cwd, "direct" if pid == _process_pid(process) else "parent"
        pid = parents.get(pid)
    return None, None


def _process_category(command: str, block_reason: str) -> str:
    lowered = command.lower()
    if "run_btc15m_overnight_shadow_monitor.py" in lowered:
        return "ai_trader_toto_mps_exporter"
    if "lean_trade.runners.run_btc_15m_tree_residual_live_quote_shadow_paper" in lowered:
        return "ai_trader_toto_worker"
    if "btc15m_shadow_overnight" in lowered and "toto" in lowered:
        return "ai_trader_toto_worker"
    if "lean_trade.runners.btc_15m_sft_shadow" in lowered:
        return "ai_trader_btc15m_sft_shadow"
    if "check_btc15m_sft_runtime_parity.py" in lowered:
        return "ai_trader_btc15m_sft_runtime_parity"
    if "verify_btc15m_activation_bank_integrity.py" in lowered:
        return "ai_trader_btc15m_activation_bank_integrity"
    if "build_btc15m_activation_rl_dataset.py" in lowered:
        return "ai_trader_btc15m_activation_rl"
    if "train_kalshi_btc15m_imitation.py" in lowered:
        return "ai_trader_btc15m_imitation"
    if "train_kalshi_btc15m_dqn.py" in lowered:
        return "ai_trader_btc15m_dqn"
    if "train_kalshi_btc15m_sft.py" in lowered:
        return "ai_trader_btc15m_sft"
    if "font_maker" in lowered and "train_node_curve_program_flow_v2.py" in lowered:
        return "font_maker_random_stroke_train"
    if "run_random_stroke_ablation_queue.py" in lowered:
        return "font_maker_random_stroke_queue"
    if "monitor_standard_glyph_exposure.py" in lowered:
        return "font_maker_standard_glyph_monitor"
    if "pytest" in lowered and "ai_trader" in lowered:
        return "ai_trader_pytest"
    if "pytest" in lowered:
        return "pytest"
    if "mds_stores" in lowered:
        return "macos_spotlight_indexer"
    if "torch" in lowered or block_reason.startswith("keyword:torch"):
        return "torch_worker"
    if block_reason.startswith("high_cpu"):
        return "high_cpu_external_job"
    return "other"


def _recent_files(root: Path, *, now_s: float, recent_seconds: float) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    matches: list[dict[str, Any]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            age_s = now_s - stat.st_mtime
            if age_s <= recent_seconds:
                try:
                    display_path = str(path.relative_to(root))
                except ValueError:
                    display_path = str(path)
                matches.append(
                    {
                        "path": display_path,
                        "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                        "age_s": round(age_s, 3),
                        "size_bytes": stat.st_size,
                    }
                )
    matches.sort(key=lambda item: float(item["age_s"]))
    return matches[:RECENT_FILE_LIMIT]


def _diagnose_process(
    process: dict[str, Any],
    *,
    now_s: float,
    recent_seconds: float,
    direct_cwds: dict[int, Path],
    parents: dict[int, int],
    live_processes: dict[int, dict[str, Any]],
    known_cwds: list[Path],
) -> dict[str, Any]:
    command = str(process.get("command") or "")
    block_reason = str(process.get("block_reason") or "")
    cwd, cwd_source = _inherited_cwd(process, direct_cwds=direct_cwds, parents=parents)
    output_dir = _resolve_output_dir(command, cwd, known_cwds)
    pid = _process_pid(process)
    live = live_processes.get(pid) if pid is not None else None
    live_pcpu = live.get("pcpu") if isinstance(live, dict) else None
    summary_pcpu = process.get("pcpu")
    pid_live = isinstance(live, dict)
    live_cpu_active = isinstance(live_pcpu, (int, float)) and float(live_pcpu) >= ACTIVE_CPU_THRESHOLD
    summary_cpu_active = isinstance(summary_pcpu, (int, float)) and float(summary_pcpu) >= ACTIVE_CPU_THRESHOLD
    cpu_blocking_threshold = _cpu_blocking_threshold(block_reason)
    recent_outputs = (
        _recent_files(output_dir, now_s=now_s, recent_seconds=recent_seconds)
        if output_dir is not None
        else []
    )
    runtime_estimate = _runtime_estimate(
        command,
        live.get("elapsed") if isinstance(live, dict) else process.get("elapsed"),
        now_s=now_s,
    )
    return {
        "pid": process.get("pid"),
        "ppid": process.get("ppid"),
        "stat": process.get("stat"),
        "elapsed": process.get("elapsed"),
        "pcpu": process.get("pcpu"),
        "pmem": process.get("pmem"),
        "block_reason": block_reason,
        "category": _process_category(command, block_reason),
        "active_cpu": live_cpu_active,
        "summary_cpu_active": summary_cpu_active,
        "live_cpu_active": live_cpu_active,
        "cpu_blocking_threshold": cpu_blocking_threshold,
        "live_cpu_over_preflight_threshold": (
            isinstance(live_pcpu, (int, float))
            and isinstance(cpu_blocking_threshold, (int, float))
            and float(live_pcpu) >= float(cpu_blocking_threshold)
        ),
        "pid_live": pid_live,
        "live_stat": live.get("stat") if isinstance(live, dict) else None,
        "live_elapsed": live.get("elapsed") if isinstance(live, dict) else None,
        "live_pcpu": live_pcpu,
        "live_pmem": live.get("pmem") if isinstance(live, dict) else None,
        "live_command": _short_text(live.get("command")) if isinstance(live, dict) else None,
        "cwd": str(cwd) if cwd is not None else None,
        "cwd_source": cwd_source,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "output_dir_exists": output_dir.exists() if output_dir is not None else None,
        "recent_output_count": len(recent_outputs),
        "recent_outputs": recent_outputs,
        **runtime_estimate,
        "command": _short_text(command),
    }


def diagnose_summary(summary_path: Path, *, recent_seconds: float = DEFAULT_RECENT_SECONDS) -> dict[str, Any]:
    summary, resolved_summary_path = _summary_payload(summary_path)
    now_s = time.time()
    processes = _processes_from_summary(summary)
    direct_cwds = _direct_cwd_by_pid(processes)
    parents = _parent_by_pid(processes)
    live_processes = _capture_live_processes(
        [pid for process in processes if (pid := _process_pid(process)) is not None]
    )
    diagnostics = [
        _diagnose_process(
            process,
            now_s=now_s,
            recent_seconds=recent_seconds,
            direct_cwds=direct_cwds,
            parents=parents,
            live_processes=live_processes,
            known_cwds=sorted(set(direct_cwds.values())),
        )
        for process in processes
    ]
    blocker_sample_count = len(diagnostics)
    blocker_count = _summary_int(summary, "preflight_blocking_process_count", default=blocker_sample_count)
    contending_sample_count = _summary_int(
        summary,
        "preflight_contending_process_sample_count",
        default=blocker_sample_count,
    )
    contending_count = _summary_int(summary, "preflight_contending_process_count", default=contending_sample_count)
    category_counts: dict[str, int] = {}
    live_category_counts: dict[str, int] = {}
    active_categories: dict[str, int] = {}
    summary_active_categories: dict[str, int] = {}
    live_cpu_blocking_categories: dict[str, int] = {}
    output_active_categories: dict[str, int] = {}
    max_remaining_s_by_category: dict[str, float] = {}
    live_or_recent_blocker_count = 0
    for item in diagnostics:
        category = str(item["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
        if item["pid_live"]:
            live_category_counts[category] = live_category_counts.get(category, 0) + 1
        if item["active_cpu"]:
            active_categories[category] = active_categories.get(category, 0) + 1
        if item["summary_cpu_active"]:
            summary_active_categories[category] = summary_active_categories.get(category, 0) + 1
        if item["live_cpu_over_preflight_threshold"]:
            live_cpu_blocking_categories[category] = live_cpu_blocking_categories.get(category, 0) + 1
        if int(item["recent_output_count"]) > 0:
            output_active_categories[category] = output_active_categories.get(category, 0) + 1
        if item["pid_live"] or int(item["recent_output_count"]) > 0:
            live_or_recent_blocker_count += 1
        remaining_s = item.get("estimated_remaining_s")
        if isinstance(remaining_s, (int, float)):
            max_remaining_s_by_category[category] = max(
                float(remaining_s),
                max_remaining_s_by_category.get(category, 0.0),
            )
    return {
        "status": "blocked" if live_or_recent_blocker_count else "no_live_or_recent_blockers_found",
        "summary_json": str(resolved_summary_path),
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "recent_seconds": float(recent_seconds),
        "process_sample_limit": summary.get("preflight_process_sample_limit"),
        "blocker_count": blocker_count,
        "blocker_sample_count": blocker_sample_count,
        "blocker_unlisted_count": max(0, blocker_count - blocker_sample_count),
        "contending_process_count": contending_count,
        "contending_process_sample_count": contending_sample_count,
        "contending_process_unlisted_count": max(0, contending_count - contending_sample_count),
        "live_blocker_count": sum(1 for item in diagnostics if item["pid_live"]),
        "recent_output_blocker_count": sum(1 for item in diagnostics if int(item["recent_output_count"]) > 0),
        "live_or_recent_blocker_count": live_or_recent_blocker_count,
        "category_counts": dict(sorted(category_counts.items())),
        "live_category_counts": dict(sorted(live_category_counts.items())),
        "active_cpu_category_counts": dict(sorted(active_categories.items())),
        "summary_cpu_active_category_counts": dict(sorted(summary_active_categories.items())),
        "live_cpu_over_preflight_threshold_category_counts": dict(sorted(live_cpu_blocking_categories.items())),
        "recent_output_category_counts": dict(sorted(output_active_categories.items())),
        "max_estimated_remaining_s_by_category": dict(sorted(max_remaining_s_by_category.items())),
        "blockers": diagnostics,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether WorldFoam MPS preflight blockers are active.")
    parser.add_argument("summary_json", type=Path, help="Launcher summary JSON, or the top-level goal audit JSON.")
    parser.add_argument("--recent-seconds", type=float, default=DEFAULT_RECENT_SECONDS)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = diagnose_summary(args.summary_json, recent_seconds=float(args.recent_seconds))
    if args.out_json is not None:
        _write_json_atomic(args.out_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
