#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


DYNAWORLD = Path(__file__).resolve().parents[2]
TRAIN_SRC = DYNAWORLD / "src" / "train"
VARIANT_ROOT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_fused_slab_v0"
VARIANT_TOOLS = VARIANT_ROOT / "tools"
RESULTS_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2" / "results"
BENCHMARK_PROCESS_COMMAND_LIMIT = 1024
BENCHMARK_PROCESS_SAMPLE_LIMIT = 32
DELTA_I16X3_FRAMEGROUP16_MODE = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE = (
    "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-materialized-fused-mse"
)
DELTA_PACKED_FRAMEGROUP16_MODE = "endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse"
DELTA_PACKED_SCALAR_MODE = "endpoint-record-delta-replace-coeff16-packed-fused-mse"
DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE = (
    "endpoint-record-delta-replace-coeff16-packed-framegroup16-materialized-fused-mse"
)
DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE = (
    "endpoint-record-delta-replace-coeff16-packed-framegroup16-recompute-fused-mse"
)
DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE = (
    "endpoint-record-delta-replace-coeff16-packed-framegroup16-smallrun16-fused-mse"
)
DELTA_AUTO_FRAMEGROUP16_MODE = "endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse"
OWNER_RUN_FUSED_MSE_MODE = "owner-run-fused-mse"
OWNER_RUN_FUSED_MSE_NOMID_MODE = "owner-run-fused-mse-nomid"
OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE = "owner-run-delta-packed-recompute-fused-mse-nomid"
OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE = (
    "owner-run-delta-packed-factorized-recompute-fused-mse-nomid"
)
OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE = (
    "owner-run-delta-packed-factorized-frameselect-recompute-fused-mse-nomid"
)
OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE = (
    "owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid"
)
OWNER_RUN_DELTA_PACKED_FACTORIZED_MODES = frozenset(
    {
        OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    }
)
ENDPOINT_RUN_FUSED_MSE_MODE = "endpoint-run-fused-mse"
GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE = "gate4-affine-candidate-num32-den16-fused-mse"
GATE4_AFFINE_CANDIDATE_NUM32_DEN16_TRACK_MSE_FUSED_MSE_MODE = (
    "gate4-affine-candidate-num32-den16-trackmse-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_FUSED_MSE_MODE = "gate4-affine-candidate-coeff16-fused-mse"
GATE4_AFFINE_CANDIDATE_COEFF16_CAP224_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-cap224-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_DENSITYMASK_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-densitymask-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_SAMPLE_REDUCE_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-samplereduce-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_SORTNET_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-sortnet-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_FRAMEGROUP16_CACHED_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-framegroup16cached-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_SITECACHE_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-sitecache-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-ownerupdate-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_I16_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-ownerupdate-i16-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-ownerkeep-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_I16_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-ownerkeep-i16-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_TRACK_MSE_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-trackmse-fused-mse"
)
GATE4_AFFINE_CANDIDATE_FUSED_MSE_MODES = frozenset(
    {
        GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_NUM32_DEN16_TRACK_MSE_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_CAP224_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_DENSITYMASK_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_SAMPLE_REDUCE_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_SORTNET_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_FRAMEGROUP16_CACHED_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_SITECACHE_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_I16_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_I16_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_TRACK_MSE_FUSED_MSE_MODE,
    }
)
DELTA_AUTO_PACKED_MAX_FRAME_COUNT = 64
DELTA_SMALLRUN16_MAX_SEGMENTS = 16
PACKED_I32_DELTA_RECORD_MODES = {
    DELTA_PACKED_SCALAR_MODE,
    DELTA_PACKED_FRAMEGROUP16_MODE,
    DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
    DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
    DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
    OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
}
ENDPOINT_SEMANTIC_TAPE_MODES = {
    "endpoint-run",
    ENDPOINT_RUN_FUSED_MSE_MODE,
    "endpoint-record-edit",
    "endpoint-record-edit-fused-mse",
    "endpoint-record-edit-coeff16-fused-mse",
    "endpoint-record-delta-replace-coeff16-fused-mse",
    "endpoint-record-delta-replace-coeff16-i16x3-fused-mse",
    DELTA_I16X3_FRAMEGROUP16_MODE,
    DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
    "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
    "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse",
    "endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse",
    DELTA_PACKED_SCALAR_MODE,
    DELTA_PACKED_FRAMEGROUP16_MODE,
    DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
    DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
    DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
    OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    DELTA_AUTO_FRAMEGROUP16_MODE,
    "endpoint-record-delta-replace-coeff16-i16x4-fused-mse",
    "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
    "endpoint-record-edit-block4",
    "endpoint-record-edit-block-coeff",
    "endpoint-record-edit-block-coeff-rgb",
    "endpoint-record-edit-block-coeff-fused-mse",
    "endpoint-record-edit-block-coeff16",
    "endpoint-record-edit-block-coeff16-fused-mse",
    "endpoint-record-edit-block-coeff16-packed-fused-mse",
    "endpoint-record-edit-block-coeff16-i16-fused-mse",
    "endpoint-record-edit-block-coeff16-i16x3-fused-mse",
    GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_NUM32_DEN16_TRACK_MSE_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_CAP224_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_DENSITYMASK_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_SAMPLE_REDUCE_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_FRAMEGROUP16_CACHED_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_SITECACHE_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_I16_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_I16_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_TRACK_MSE_FUSED_MSE_MODE,
}


def _is_gate4_affine_candidate_fused_mse_mode(tape_mode: str) -> bool:
    return tape_mode in GATE4_AFFINE_CANDIDATE_FUSED_MSE_MODES


def _is_gate4_affine_candidate_coeff16_mode(tape_mode: str) -> bool:
    return tape_mode in {
        GATE4_AFFINE_CANDIDATE_COEFF16_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_CAP224_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_DENSITYMASK_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_SAMPLE_REDUCE_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_SORTNET_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_FRAMEGROUP16_CACHED_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_SITECACHE_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_I16_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_I16_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_TRACK_MSE_FUSED_MSE_MODE,
    }


def _is_gate4_affine_candidate_sample_reduce_mode(tape_mode: str) -> bool:
    return tape_mode == GATE4_AFFINE_CANDIDATE_COEFF16_SAMPLE_REDUCE_FUSED_MSE_MODE


def _is_gate4_affine_candidate_cap224_mode(tape_mode: str) -> bool:
    return tape_mode == GATE4_AFFINE_CANDIDATE_COEFF16_CAP224_FUSED_MSE_MODE


def _is_gate4_affine_candidate_densitymask_mode(tape_mode: str) -> bool:
    return tape_mode == GATE4_AFFINE_CANDIDATE_COEFF16_DENSITYMASK_FUSED_MSE_MODE


def _is_gate4_affine_candidate_sortnet_mode(tape_mode: str) -> bool:
    return tape_mode == GATE4_AFFINE_CANDIDATE_COEFF16_SORTNET_FUSED_MSE_MODE


def _is_gate4_affine_candidate_framegroup16_cached_mode(tape_mode: str) -> bool:
    return tape_mode == GATE4_AFFINE_CANDIDATE_COEFF16_FRAMEGROUP16_CACHED_FUSED_MSE_MODE


def _is_gate4_affine_candidate_sitecache_mode(tape_mode: str) -> bool:
    return tape_mode == GATE4_AFFINE_CANDIDATE_COEFF16_SITECACHE_FUSED_MSE_MODE


def _is_gate4_affine_candidate_ownerupdate_mode(tape_mode: str) -> bool:
    return tape_mode == GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_FUSED_MSE_MODE


def _is_gate4_affine_candidate_ownerupdate_i16_mode(tape_mode: str) -> bool:
    return tape_mode == GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_I16_FUSED_MSE_MODE


def _is_gate4_affine_candidate_ownerkeep_mode(tape_mode: str) -> bool:
    return tape_mode == GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_FUSED_MSE_MODE


def _is_gate4_affine_candidate_ownerkeep_i16_mode(tape_mode: str) -> bool:
    return tape_mode == GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_I16_FUSED_MSE_MODE


def _is_gate4_affine_candidate_trackmse_mode(tape_mode: str) -> bool:
    return tape_mode in {
        GATE4_AFFINE_CANDIDATE_NUM32_DEN16_TRACK_MSE_FUSED_MSE_MODE,
        GATE4_AFFINE_CANDIDATE_COEFF16_TRACK_MSE_FUSED_MSE_MODE,
    }


def _resolve_delta_framegroup16_auto_mode(
    tape_mode: str,
    *,
    frame_count: int,
    prefer_smallrun16: bool = False,
) -> str:
    if tape_mode != DELTA_AUTO_FRAMEGROUP16_MODE:
        return tape_mode
    if frame_count <= DELTA_AUTO_PACKED_MAX_FRAME_COUNT:
        if prefer_smallrun16:
            return DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE
        return DELTA_PACKED_FRAMEGROUP16_MODE
    return DELTA_I16X3_FRAMEGROUP16_MODE


def _delta_mode_uses_packed_i32_records(tape_mode: str) -> bool:
    return tape_mode in PACKED_I32_DELTA_RECORD_MODES


def _delta_mode_uses_packed_framegroup(tape_mode: str) -> bool:
    return tape_mode in {
        DELTA_PACKED_FRAMEGROUP16_MODE,
        DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
        DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
        DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
        OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    }


def _effective_native_emitted_pack_records(*, requested: bool, resolved_tape_mode: str) -> bool:
    return bool(requested) and _delta_mode_uses_packed_i32_records(resolved_tape_mode)


def _delta_framegroup_record_bytes(tape_mode: str) -> int:
    if _delta_mode_uses_packed_i32_records(tape_mode):
        return 4
    if "i16x4" in tape_mode:
        return 8
    return 6


def _benchmark_keyword_matches(command: str, keyword: str) -> bool:
    if keyword == "python":
        return keyword in command
    return re.search(rf"(^|[^a-z0-9]){re.escape(keyword)}([^a-z0-9]|$)", command) is not None


def _benchmark_process_blocks_promotion(
    *,
    command: str,
    pcpu: float,
    blocking_cpu_threshold: float,
    general_blocking_cpu_threshold: float,
    hard_keywords: tuple[str, ...],
) -> bool:
    return _benchmark_process_block_reason(
        command=command,
        pcpu=pcpu,
        blocking_cpu_threshold=blocking_cpu_threshold,
        general_blocking_cpu_threshold=general_blocking_cpu_threshold,
        hard_keywords=hard_keywords,
    ) is not None


def _benchmark_process_block_reason(
    *,
    command: str,
    pcpu: float,
    blocking_cpu_threshold: float,
    general_blocking_cpu_threshold: float,
    hard_keywords: tuple[str, ...],
) -> str | None:
    lowered = command.lower()
    cpu_keyword_matched = any(
        _benchmark_keyword_matches(lowered, keyword)
        for keyword in ("python", "pytest", "torch", "metal", "mps", "modal")
    )
    if pcpu >= blocking_cpu_threshold and cpu_keyword_matched:
        return "high_cpu"
    if pcpu >= general_blocking_cpu_threshold:
        return "high_cpu_general"
    if _benchmark_periodic_mps_exporter_blocks_promotion(lowered):
        return "periodic_mps_exporter"
    if _benchmark_low_cpu_wrapper_is_background(lowered):
        return None
    for keyword in hard_keywords:
        if keyword != "pytest" and _benchmark_keyword_matches(lowered, keyword):
            return f"keyword:{keyword}"
    return None


def _benchmark_periodic_mps_exporter_blocks_promotion(command: str) -> bool:
    return (
        "run_btc15m_overnight_shadow_monitor.py" in command
        and ("--toto-export-device mps" in command or "--toto-export-with-runtime-deps" in command)
    )


def _benchmark_low_cpu_wrapper_is_background(command: str) -> bool:
    wrapper_markers = (
        "screen -dms",
        "login -pflq",
        "run_btc15m_overnight_shadow_monitor.py",
        "run_worldfoam_clean_mps_gate_when_ready.py",
        "run_worldfoam_star_native_cutwalk_gate.py",
        "summarize_btc15m_overnight",
        "watch_schema.py --watch",
        "sky.server.server",
        "mtlcompilerservice",
    )
    return any(marker in command for marker in wrapper_markers)


def _benchmark_ignored_process_pids(
    *,
    own_pid: int,
    own_ppid: int,
    parent_by_pid: dict[int, int],
) -> set[int]:
    ignored_pids = {own_pid}
    parent_pid = parent_by_pid.get(own_pid, own_ppid)
    while parent_pid > 1 and parent_pid not in ignored_pids:
        ignored_pids.add(parent_pid)
        next_parent_pid = parent_by_pid.get(parent_pid)
        if next_parent_pid is None:
            break
        parent_pid = next_parent_pid
    return ignored_pids


def _parse_benchmark_ps_line(line: str) -> dict[str, Any] | None:
    parts = line.split(None, 6)
    if len(parts) >= 7:
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
            pass

    legacy_parts = line.split(None, 4)
    if len(legacy_parts) < 5:
        return None
    try:
        return {
            "pid": int(legacy_parts[0]),
            "ppid": int(legacy_parts[1]),
            "stat": "",
            "elapsed": "",
            "pcpu": float(legacy_parts[2]),
            "pmem": float(legacy_parts[3]),
            "command": legacy_parts[4],
        }
    except ValueError:
        return None


def _capture_benchmark_environment() -> dict[str, Any]:
    keywords = ("python", "pytest", "torch", "metal", "mps", "modal")
    hard_keywords = ("pytest", "torch", "metal", "mps")
    blocking_cpu_threshold = 5.0
    general_blocking_cpu_threshold = 75.0
    own_pid = os.getpid()
    own_ppid = os.getppid()
    try:
        result = subprocess.run(
            ["ps", "-wwaxo", "pid=,ppid=,stat=,etime=,pcpu=,pmem=,command="],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "unchecked",
            "error": str(exc),
            "pid": own_pid,
            "keywords": list(keywords),
            "hard_keywords": list(hard_keywords),
            "blocking_cpu_threshold": blocking_cpu_threshold,
            "general_blocking_cpu_threshold": general_blocking_cpu_threshold,
            "blocking_process_count": 0,
            "contending_process_count": 0,
            "background_process_count": 0,
            "process_sample_limit": BENCHMARK_PROCESS_SAMPLE_LIMIT,
            "blocking_processes": [],
            "background_processes": [],
            "contending_processes": [],
        }

    process_rows: list[dict[str, Any]] = []
    parent_by_pid: dict[int, int] = {}
    for line in result.stdout.splitlines():
        process_row = _parse_benchmark_ps_line(line)
        if process_row is None:
            continue
        pid = int(process_row["pid"])
        ppid = int(process_row["ppid"])
        process_rows.append(process_row)
        parent_by_pid[pid] = ppid

    ignored_pids = _benchmark_ignored_process_pids(
        own_pid=own_pid,
        own_ppid=own_ppid,
        parent_by_pid=parent_by_pid,
    )
    blocking_processes: list[dict[str, Any]] = []
    background_processes: list[dict[str, Any]] = []
    for process_row in process_rows:
        pid = int(process_row["pid"])
        if pid in ignored_pids:
            continue
        ppid = int(process_row["ppid"])
        pcpu = float(process_row["pcpu"])
        pmem = float(process_row["pmem"])
        command = str(process_row["command"])
        lowered = command.lower()
        keyword_matched = any(_benchmark_keyword_matches(lowered, keyword) for keyword in keywords)
        if not keyword_matched and pcpu < general_blocking_cpu_threshold:
            continue
        process = {
            "pid": pid,
            "ppid": ppid,
            "stat": str(process_row.get("stat") or ""),
            "elapsed": str(process_row.get("elapsed") or ""),
            "pcpu": pcpu,
            "pmem": pmem,
            "command": command[:BENCHMARK_PROCESS_COMMAND_LIMIT],
        }
        block_reason = _benchmark_process_block_reason(
            command=command,
            pcpu=pcpu,
            blocking_cpu_threshold=blocking_cpu_threshold,
            general_blocking_cpu_threshold=general_blocking_cpu_threshold,
            hard_keywords=hard_keywords,
        )
        if block_reason is not None:
            process["block_reason"] = block_reason
            blocking_processes.append(process)
        else:
            background_processes.append(process)
    blocking_processes.sort(key=lambda item: float(item["pcpu"]), reverse=True)
    background_processes.sort(key=lambda item: float(item["pcpu"]), reverse=True)
    status = "contended" if blocking_processes else ("background" if background_processes else "ok")
    return {
        "status": status,
        "pid": own_pid,
        "keywords": list(keywords),
        "hard_keywords": list(hard_keywords),
        "blocking_cpu_threshold": blocking_cpu_threshold,
        "general_blocking_cpu_threshold": general_blocking_cpu_threshold,
        "blocking_process_count": len(blocking_processes),
        "contending_process_count": len(blocking_processes),
        "background_process_count": len(background_processes),
        "process_sample_limit": BENCHMARK_PROCESS_SAMPLE_LIMIT,
        "blocking_processes": blocking_processes[:BENCHMARK_PROCESS_SAMPLE_LIMIT],
        "background_processes": background_processes[:BENCHMARK_PROCESS_SAMPLE_LIMIT],
        "contending_processes": blocking_processes[:BENCHMARK_PROCESS_SAMPLE_LIMIT],
    }


def _merge_benchmark_environments(start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    status = "contended" if "contended" in {start.get("status"), end.get("status")} else start.get("status", "unchecked")
    if status == "ok" and end.get("status") != "ok":
        status = str(end.get("status", "unchecked"))
    return {
        "status": status,
        "start": start,
        "end": end,
    }


def _is_mtl_compiler_process(process: dict[str, Any]) -> bool:
    command = str(process.get("command", "")).lower()
    return "mtlcompilerservice" in command


def _only_mtl_compiler_blocks(environment: dict[str, Any]) -> bool:
    blocking = environment.get("blocking_processes")
    if environment.get("status") != "contended" or not isinstance(blocking, list) or not blocking:
        return False
    return all(isinstance(process, dict) and _is_mtl_compiler_process(process) for process in blocking)


def _merge_benchmark_environments_with_optional_settle(
    start: dict[str, Any],
    *,
    settle_s: float,
) -> dict[str, Any]:
    immediate_end = _capture_benchmark_environment()
    if settle_s <= 0.0 or not _only_mtl_compiler_blocks(immediate_end):
        return _merge_benchmark_environments(start, immediate_end)

    time.sleep(float(settle_s))
    settled_end = _capture_benchmark_environment()
    merged = _merge_benchmark_environments(start, settled_end)
    merged["end_immediate"] = immediate_end
    merged["post_run_settle_s"] = float(settle_s)
    merged["transient_mtl_compiler_settled"] = merged["status"] != "contended"
    return merged


def _benchmark_environment_blocks_promotion(environment: dict[str, Any]) -> bool:
    return environment.get("status") not in {"ok", "background"}


def _wait_for_benchmark_environment_ok(*, timeout_s: float, poll_s: float) -> dict[str, Any]:
    started_at = time.perf_counter()
    environment = _capture_benchmark_environment()
    while _benchmark_environment_blocks_promotion(environment):
        if timeout_s <= 0.0 or time.perf_counter() - started_at >= timeout_s:
            return environment
        print(
            "benchmark environment contended; waiting for a quiet window",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(max(0.1, poll_s))
        environment = _capture_benchmark_environment()
    return environment


def _print_benchmark_environment(environment: dict[str, Any]) -> None:
    print(json.dumps(environment, indent=2, sort_keys=True), flush=True)

for path in (TRAIN_SRC, VARIANT_ROOT, VARIANT_TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gate4_moving_ray_slab_compiler import (  # noqa: E402
    DEFAULT_CONFIG,
    SITE_INITIALIZATION_CHOICES,
    SITE_INITIALIZATION_LEGACY_SPARSE,
    SyntheticRayMotion,
    _load_config,
    apply_synthetic_ray_motion,
    fit_linear_ray_track,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
)
from gate4_affine_slab_tape import (  # noqa: E402
    Gate4EndpointDeltaReplaceTape,
    build_gate4_affine_slab_tape,
    build_gate4_boundary_depth_coefficients,
    build_gate4_endpoint_delta_replace_tape,
    build_gate4_endpoint_run_sequences,
)
from probe_fused_slab_segment_tape import SegmentTape, build_segment_tape, compact_segment_tape  # noqa: E402
from probe_endpoint_record_delta_replay import (  # noqa: E402
    _boundary_tensor,
    _track_frame_rays,
    build_delta_replace_chunk_change_offsets,
    build_delta_replace_chunk_owner_lists,
    build_delta_replace_frame_row_descriptors,
    pack_endpoint_record_delta_replace_tape,
)
from probe_endpoint_record_edit_replay import (  # noqa: E402
    _track_boundary_coefficients,
    pack_endpoint_record_block_edit_tape,
    pack_endpoint_record_edit_tape,
)
from probe_endpoint_run_tape import compress_same_owner_endpoint_runs  # noqa: E402
from probe_owner_run_boundary_tape import _build_owner_run_sequences  # noqa: E402
from probe_segment_owner_run_tape import compress_same_owner_runs  # noqa: E402
from smoke_fused_slab_affine_realray_mps import _parse_int_list  # noqa: E402
from objective.world_foam_frozen_rgb_mse import (  # noqa: E402
    FRAMEGROUP16_TAPE_KEYS,
    WorldFoamFrozenRGBMSEObjective,
    WorldFoamTargetLayout,
    promoted_framegroup16_loss_fn,
)
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    RealRayReplayConfig,
    endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_delta_replace_rgba_depth_replay,
    endpoint_record_edit_block_coeff16_rgba_depth_replay,
    endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block_coeff_rgba_depth_autograd,
    endpoint_record_edit_block_coeff_rgba_depth_replay,
    endpoint_record_edit_block_coeff_rgb_autograd,
    endpoint_record_edit_block_coeff_rgb_replay,
    endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_block4_rgba_depth_autograd,
    endpoint_record_edit_block4_rgba_depth_replay,
    endpoint_record_edit_block4_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_rgba_depth_autograd,
    endpoint_record_edit_rgba_depth_replay,
    endpoint_record_edit_mse_vjp_direct_atomic_rgb_only,
    endpoint_record_edit_vjp_direct_atomic_grad_only,
    endpoint_record_edit_vjp_direct_atomic_rgb_only,
    endpoint_run_mse_vjp_direct_atomic_rgb_only,
    endpoint_run_rgba_depth_autograd,
    endpoint_run_rgba_depth_replay,
    endpoint_run_vjp_direct_atomic_grad_only,
    fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only,
    fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only,
    fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only,
    fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only,
    fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only,
    fused_slab_affine_coeff16_realray_rgba_depth_replay,
    fused_slab_affine_num32_den16_realray_rgba_depth_replay,
    segment_tape_mse_vjp_direct_atomic_rgb_only,
    segment_tape_nomids_mse_vjp_direct_atomic_rgb_only,
    segment_tape_rgba_depth_autograd,
    segment_tape_rgba_depth_replay,
    segment_tape_vjp_direct_atomic_grad_only,
)


def _psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1.0e-12))


def _metrics(rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    rendered = rendered.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
    target = target.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
    mse = float(torch.mean((rendered - target).square()).cpu().item())
    return {
        "mse": mse,
        "psnr": _psnr_from_mse(mse),
        "l1": float(torch.mean(torch.abs(rendered - target)).cpu().item()),
    }


def _ratio_last_first(last: float, first: float) -> float:
    if abs(float(first)) <= 1.0e-12:
        return 0.0 if abs(float(last)) <= 1.0e-12 else float("inf")
    return float(last) / float(first)


def _world_foam_objective_adapter_metadata(objective: WorldFoamFrozenRGBMSEObjective | None) -> dict[str, Any] | None:
    if objective is None:
        return None
    scope = objective.scope
    return {
        "name": type(objective).__name__,
        "module": type(objective).__module__,
        "construction_scope": "once_per_frame_count_run",
        "loss_call_scope": "per_optimizer_step",
        "backend_loss_fn": "promoted_framegroup16_loss_fn",
        "tape_mode": scope.tape_mode,
        "gradient_scope": scope.gradient_scope,
        "full_trainer_claim": scope.full_trainer_claim,
        "full_geometry_gradient_claim": scope.full_geometry_gradient_claim,
        "quality_claim": scope.quality_claim,
        "renderer_backend_claim": scope.renderer_backend_claim,
        "supports_rgb_mse_only": scope.supports_rgb_mse_only,
        "supports_background_composition": scope.supports_background_composition,
        "supports_colorizer": scope.supports_colorizer,
        "supports_vjepa_feature_loss": scope.supports_vjepa_feature_loss,
    }


def _image_rgb_from_track_major(
    rgb: torch.Tensor,
    *,
    view_count: int,
    frame_count: int,
    height: int,
    width: int,
) -> torch.Tensor:
    return (
        rgb.reshape(view_count, height, width, frame_count, 3)
        .permute(0, 3, 4, 1, 2)
        .reshape(view_count * frame_count, 3, height, width)
    )


def _track_major_rgb_from_image(
    rgb_image: torch.Tensor,
    *,
    view_count: int,
    frame_count: int,
    height: int,
    width: int,
) -> torch.Tensor:
    return (
        rgb_image.reshape(view_count, frame_count, 3, height, width)
        .permute(0, 3, 4, 1, 2)
        .reshape(view_count * height * width, frame_count, 3)
        .contiguous()
    )


def _track_ray_linear_coefficients(*, track_rays: torch.Tensor, frame_t: torch.Tensor) -> torch.Tensor:
    if track_rays.ndim != 3 or track_rays.shape[2] != 6:
        raise ValueError("track_rays must have shape [track_count, frame_count, 6]")
    times = frame_t.to(dtype=torch.float64).cpu()
    rows: list[tuple[float, ...]] = []
    for track_id in range(int(track_rays.shape[0])):
        track = fit_linear_ray_track(track_rays[track_id].cpu(), times)
        rows.append(
            (
                *track.origin_base,
                *track.origin_slope,
                *track.direction_base,
                *track.direction_slope,
            )
        )
    return torch.tensor(rows, dtype=torch.float32)


def _track_major_grad_from_image(
    grad_rgb_image: torch.Tensor,
    *,
    view_count: int,
    frame_count: int,
    height: int,
    width: int,
) -> torch.Tensor:
    return _track_major_rgb_from_image(
        grad_rgb_image,
        view_count=view_count,
        frame_count=frame_count,
        height=height,
        width=width,
    )


def _repeat_view_major_frames(
    values: torch.Tensor,
    *,
    loaded_frame_count: int,
    requested_frame_count: int,
    name: str,
) -> torch.Tensor:
    if loaded_frame_count < 1 or requested_frame_count < 1:
        raise ValueError(f"{name} frame counts must be positive")
    sample_count = int(values.shape[0])
    if sample_count % loaded_frame_count != 0:
        raise ValueError(
            f"{name} sample count {sample_count} is not divisible by loaded_frame_count={loaded_frame_count}"
        )
    if loaded_frame_count == requested_frame_count:
        return values
    view_count = sample_count // loaded_frame_count
    source_frame_indices = (
        torch.arange(requested_frame_count, dtype=torch.long, device=values.device) % loaded_frame_count
    )
    return (
        values.reshape(view_count, loaded_frame_count, *values.shape[1:])
        .index_select(1, source_frame_indices)
        .reshape(view_count * requested_frame_count, *values.shape[1:])
        .contiguous()
    )


def _sequential_view_major_frame_indices(
    *,
    view_count: int,
    requested_frame_count: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.arange(requested_frame_count, dtype=torch.long, device=device).repeat(view_count)


def _fit_loaded_frame_count(
    *,
    split_name: str,
    targets: torch.Tensor,
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    loaded_frame_count: int,
    requested_frame_count: int,
    allow_repeat_loaded_frames: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if loaded_frame_count == requested_frame_count:
        return targets, rays, frame_indices, False
    if loaded_frame_count > requested_frame_count:
        raise ValueError(
            f"{split_name} loader returned {loaded_frame_count} frames for requested "
            f"{requested_frame_count}; expected the data loader to crop to the requested count"
        )
    if not allow_repeat_loaded_frames:
        raise ValueError(
            f"{split_name} loader returned only {loaded_frame_count} frames for requested "
            f"{requested_frame_count}; pass --repeat-loaded-frames for a synthetic repeated-fixture "
            "speed-scaling smoke, or use a longer real multicam fixture"
        )
    if int(targets.shape[0]) % loaded_frame_count != 0 or int(rays.shape[0]) % loaded_frame_count != 0:
        raise ValueError(f"{split_name} tensors are not view-major by loaded_frame_count={loaded_frame_count}")
    targets = _repeat_view_major_frames(
        targets,
        loaded_frame_count=loaded_frame_count,
        requested_frame_count=requested_frame_count,
        name=f"{split_name}.targets",
    )
    rays = _repeat_view_major_frames(
        rays,
        loaded_frame_count=loaded_frame_count,
        requested_frame_count=requested_frame_count,
        name=f"{split_name}.rays",
    )
    view_count = int(targets.shape[0]) // requested_frame_count
    frame_indices = _sequential_view_major_frame_indices(
        view_count=view_count,
        requested_frame_count=requested_frame_count,
        device=frame_indices.device,
    )
    return targets, rays, frame_indices, True


def _summarize_steps(rows: list[dict[str, float]]) -> dict[str, dict[str, float | int]]:
    keys = sorted({key for row in rows for key in row if key != "loss"})
    out: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = sorted(float(row[key]) for row in rows if key in row)
        p90_index = min(len(values) - 1, math.ceil(0.90 * len(values)) - 1)
        out[key] = {
            "count": len(values),
            "mean_s": statistics.fmean(values),
            "median_s": statistics.median(values),
            "p90_s": values[p90_index],
            "min_s": min(values),
            "max_s": max(values),
            "max_to_median_ratio": max(values) / max(statistics.median(values), 1.0e-12),
        }
    return out


def _move_tape_to_mps(tape: Any, *, include_mids: bool = True) -> dict[str, torch.Tensor]:
    device = torch.device("mps")
    out = {
        "offsets_i32": tape.offsets_i32.to(device=device).contiguous(),
        "owners_i32": tape.owners_i32.to(device=device).contiguous(),
        "lengths_f32": tape.lengths_f32.to(device=device).contiguous(),
    }
    if include_mids:
        out["mids_f32"] = tape.mids_f32.to(device=device).contiguous()
    return out


def _move_endpoint_tape_to_mps(tape: Any) -> dict[str, torch.Tensor]:
    device = torch.device("mps")
    return {
        "offsets_i32": tape.offsets_i32.to(device=device).contiguous(),
        "owners_i32": tape.owners_i32.to(device=device).contiguous(),
        "starts_f32": tape.starts_f32.to(device=device).contiguous(),
        "ends_f32": tape.ends_f32.to(device=device).contiguous(),
    }


def _move_endpoint_record_edit_tape_to_mps(
    *,
    edit: Any,
    boundary_f32: torch.Tensor,
    rays_f32: torch.Tensor,
    frame_t_f32: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = torch.device("mps")
    return {
        "boundary_f32": boundary_f32.to(device=device).contiguous(),
        "rays_f32": rays_f32.to(device=device).contiguous(),
        "frame_t_f32": frame_t_f32.to(device=device).contiguous(),
        "base_offsets_i32": edit.base_offsets_i32.to(device=device).contiguous(),
        "base_owner_i32": edit.base_owner_i32.to(device=device).contiguous(),
        "base_left_i32": edit.base_left_i32.to(device=device).contiguous(),
        "base_right_i32": edit.base_right_i32.to(device=device).contiguous(),
        "track_change_offsets_i32": edit.track_change_offsets_i32.to(device=device).contiguous(),
        "change_frame_i32": edit.change_frame_i32.to(device=device).contiguous(),
        "op_offsets_i32": edit.op_offsets_i32.to(device=device).contiguous(),
        "op_type_i32": edit.op_type_i32.to(device=device).contiguous(),
        "op_pos_i32": edit.op_pos_i32.to(device=device).contiguous(),
        "op_owner_i32": edit.op_owner_i32.to(device=device).contiguous(),
        "op_left_i32": edit.op_left_i32.to(device=device).contiguous(),
        "op_right_i32": edit.op_right_i32.to(device=device).contiguous(),
    }


def _move_endpoint_record_delta_replace_tape_to_mps(
    *,
    delta: Any,
    boundary_f32: torch.Tensor,
    rays_f32: torch.Tensor,
    frame_t_f32: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = torch.device("mps")
    delta_tables = _validate_endpoint_delta_index_tables(delta)
    return {
        "boundary_f32": boundary_f32.to(device=device).contiguous(),
        "rays_f32": rays_f32.to(device=device).contiguous(),
        "frame_t_f32": frame_t_f32.to(device=device).contiguous(),
        "base_offsets_i32": delta_tables["base_offsets_i32"].to(device=device).contiguous(),
        "base_owner_i32": delta.base_owner_i32.to(device=device).contiguous(),
        "base_left_i32": delta.base_left_i32.to(device=device).contiguous(),
        "base_right_i32": delta.base_right_i32.to(device=device).contiguous(),
        "track_change_offsets_i32": delta_tables["track_change_offsets_i32"].to(device=device).contiguous(),
        "change_frame_i32": delta_tables["change_frame_i32"].to(device=device).contiguous(),
        "change_offsets_i32": delta_tables["change_offsets_i32"].to(device=device).contiguous(),
        "change_owner_i32": delta.change_owner_i32.to(device=device).contiguous(),
        "change_left_i32": delta.change_left_i32.to(device=device).contiguous(),
        "change_right_i32": delta.change_right_i32.to(device=device).contiguous(),
    }


_MINIMAL_DELTA_FUSED_DEVICE_TENSOR_KEYS = (
    "frame_t_f32",
    "base_offsets_i32",
    "track_change_offsets_i32",
    "change_frame_i32",
    "change_offsets_i32",
)


def _move_endpoint_record_delta_replace_minimal_fused_tape_to_mps(
    *,
    delta: Any,
    frame_t_f32: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = torch.device("mps")
    delta_tables = _validate_endpoint_delta_index_tables(delta)
    tensors = {
        "frame_t_f32": frame_t_f32.to(device=device).contiguous(),
        "base_offsets_i32": delta_tables["base_offsets_i32"].to(device=device).contiguous(),
        "track_change_offsets_i32": delta_tables["track_change_offsets_i32"].to(device=device).contiguous(),
        "change_frame_i32": delta_tables["change_frame_i32"].to(device=device).contiguous(),
        "change_offsets_i32": delta_tables["change_offsets_i32"].to(device=device).contiguous(),
    }
    return {key: tensors[key] for key in _MINIMAL_DELTA_FUSED_DEVICE_TENSOR_KEYS}


def _move_gate4_affine_candidate_tape_to_mps(
    *,
    gate4_tape: Any,
    sites: tuple[Any, ...],
    trackmse_fused_mse: bool = False,
    coeff16_fused_mse: bool = False,
    cap224_fused_mse: bool = False,
    densitymask_fused_mse: bool = False,
    sample_reduce_fused_mse: bool = False,
    sortnet_fused_mse: bool = False,
    framegroup16_cached_fused_mse: bool = False,
    sitecache_fused_mse: bool = False,
    ownerupdate_fused_mse: bool = False,
    ownerupdate_i16_fused_mse: bool = False,
    ownerkeep_fused_mse: bool = False,
    ownerkeep_i16_fused_mse: bool = False,
) -> dict[str, Any]:
    device = torch.device("mps")
    boundary_pairs = [(int(boundary.left), int(boundary.right)) for boundary in make_boundaries_4d(sites)]
    tensors: dict[str, Any] = {
        "affine_row_index_i32": gate4_tape.row_index.to(device=device, dtype=torch.int32).contiguous(),
        "affine_candidate_row_offsets_i32": gate4_tape.row_offsets.to(device=device, dtype=torch.int32).contiguous(),
        "affine_sites_f32": torch.tensor(
            [[site.x, site.y, site.z, site.t, site.weight] for site in sites],
            dtype=torch.float32,
            device=device,
        ).contiguous(),
        "affine_ray_f32": gate4_tape.ray_coeff.to(device=device, dtype=torch.float32).contiguous(),
        "affine_frame_t_f32": gate4_tape.frame_t.to(device=device, dtype=torch.float32).contiguous(),
        "affine_candidate_fused_mse": True,
        "affine_candidate_trackmse_fused_mse": bool(trackmse_fused_mse),
        "affine_candidate_coeff16_fused_mse": bool(coeff16_fused_mse),
        "affine_candidate_cap224_fused_mse": bool(cap224_fused_mse),
        "affine_candidate_densitymask_fused_mse": bool(densitymask_fused_mse),
        "affine_candidate_sample_reduce_fused_mse": bool(sample_reduce_fused_mse),
        "affine_candidate_sortnet_fused_mse": bool(sortnet_fused_mse),
        "affine_candidate_framegroup16_cached_fused_mse": bool(framegroup16_cached_fused_mse),
        "affine_candidate_sitecache_fused_mse": bool(sitecache_fused_mse),
        "affine_candidate_ownerupdate_fused_mse": bool(ownerupdate_fused_mse),
        "affine_candidate_ownerupdate_i16_fused_mse": bool(ownerupdate_i16_fused_mse),
        "affine_candidate_ownerkeep_fused_mse": bool(ownerkeep_fused_mse),
        "affine_candidate_ownerkeep_i16_fused_mse": bool(ownerkeep_i16_fused_mse),
        "affine_time_slab_count": int(gate4_tape.time_slab_count),
        "affine_row_count": int(gate4_tape.row_count),
    }
    if ownerupdate_fused_mse or ownerkeep_fused_mse:
        tensors["affine_candidate_boundary_ids_i32"] = gate4_tape.candidate_ids.to(
            device=device,
            dtype=torch.int32,
        ).contiguous()
        tensors["affine_boundary_site_pairs_i32"] = (
            torch.tensor(boundary_pairs, dtype=torch.int32, device=device)
            if boundary_pairs
            else torch.empty((0, 2), dtype=torch.int32, device=device)
        ).contiguous()
    if ownerupdate_i16_fused_mse or ownerkeep_i16_fused_mse:
        if len(boundary_pairs) > 32767 or len(sites) > 32767:
            raise ValueError("ownerupdate/ownerkeep-i16 requires boundary and site counts to fit int16")
        tensors["affine_candidate_boundary_ids_i16"] = gate4_tape.candidate_ids.to(
            device=device,
            dtype=torch.int16,
        ).contiguous()
        tensors["affine_boundary_site_pairs_i16"] = (
            torch.tensor(boundary_pairs, dtype=torch.int16, device=device)
            if boundary_pairs
            else torch.empty((0, 2), dtype=torch.int16, device=device)
        ).contiguous()
    if coeff16_fused_mse:
        tensors["affine_candidate_depth_coeff_f16"] = gate4_tape.candidate_depth_coeffs.to(
            device=device,
            dtype=torch.float16,
        ).contiguous()
    else:
        tensors["affine_candidate_depth_num_f32"] = gate4_tape.candidate_depth_num.to(
            device=device,
            dtype=torch.float32,
        ).contiguous()
        tensors["affine_candidate_depth_den_f16"] = gate4_tape.candidate_depth_den().to(device=device).contiguous()
    return tensors


def _rebase_endpoint_record_delta_replace_cpu(delta: Any) -> Any:
    return type(delta)(
        base_offsets_i32=delta.base_offsets_i32.detach().clone().contiguous(),
        base_owner_i32=delta.base_owner_i32.detach().clone().contiguous(),
        base_left_i32=delta.base_left_i32.detach().clone().contiguous(),
        base_right_i32=delta.base_right_i32.detach().clone().contiguous(),
        track_change_offsets_i32=delta.track_change_offsets_i32.detach().clone().contiguous(),
        change_frame_i32=delta.change_frame_i32.detach().clone().contiguous(),
        change_offsets_i32=delta.change_offsets_i32.detach().clone().contiguous(),
        change_owner_i32=delta.change_owner_i32.detach().clone().contiguous(),
        change_left_i32=delta.change_left_i32.detach().clone().contiguous(),
        change_right_i32=delta.change_right_i32.detach().clone().contiguous(),
        base_record_i32=delta.base_record_i32.detach().clone().contiguous()
        if getattr(delta, "base_record_i32", None) is not None
        else None,
        change_record_i32=delta.change_record_i32.detach().clone().contiguous()
        if getattr(delta, "change_record_i32", None) is not None
        else None,
    )


def _render_device_for_tape(tape_info: dict[str, Any]) -> dict[str, torch.Tensor]:
    selected_device = tape_info["selected_device"]
    if "base_owner_i32" in selected_device:
        return selected_device
    if "lengths_f32" in selected_device and "mids_f32" not in selected_device:
        render_device = tape_info.get("segment_render_device")
        if render_device is None:
            selected = tape_info.get("selected")
            if selected is None:
                raise RuntimeError("segment render device requires the CPU selected tape with mids")
            render_device = _move_tape_to_mps(selected, include_mids=True)
            tape_info["segment_render_device"] = render_device
        return render_device
    render_device = tape_info.get("endpoint_record_delta_render_device")
    if render_device is not None:
        return render_device
    delta = tape_info.get("endpoint_record_delta_replace")
    render_inputs = tape_info.get("endpoint_record_delta_render_inputs")
    if delta is None or render_inputs is None:
        return selected_device
    render_device = _move_endpoint_record_delta_replace_tape_to_mps(
        delta=delta,
        boundary_f32=render_inputs["boundary_f32"],
        rays_f32=render_inputs["rays_f32"],
        frame_t_f32=render_inputs["frame_t_f32"],
    )
    tape_info["endpoint_record_delta_render_device"] = render_device
    return render_device


def _pack_cut_ids_i32(cut_i32: torch.Tensor) -> torch.Tensor:
    cut = cut_i32.detach().cpu().to(dtype=torch.int64)
    invalid_negative = (cut < -2).any()
    if bool(invalid_negative.item()):
        raise ValueError("packed endpoint records only support cut ids -1, -2, or nonnegative boundary ids")
    code = torch.where(cut == -1, torch.zeros_like(cut), torch.where(cut == -2, torch.ones_like(cut), cut + 2))
    if code.numel() and int(code.max().item()) > 4095:
        raise ValueError("packed endpoint records support cut codes up to 4095")
    return code


def _pack_endpoint_records_i32_native_op() -> Any | None:
    try:
        return torch.ops.world_foam_lane2_fused_slab_v0.pack_endpoint_records_i32_cpu
    except (AttributeError, RuntimeError):
        return None


def _pack_endpoint_records_i32(
    owner_i32: torch.Tensor,
    left_i32: torch.Tensor,
    right_i32: torch.Tensor,
    *,
    use_native: bool = False,
    site_count: int | None = None,
    boundary_count: int | None = None,
) -> torch.Tensor:
    records = _validate_endpoint_record_components(
        "packed endpoint record",
        owner_i32=owner_i32,
        left_i32=left_i32,
        right_i32=right_i32,
        site_count=site_count,
        boundary_count=boundary_count,
    )
    owner_i32 = records["owner_i32"]
    left_i32 = records["left_i32"]
    right_i32 = records["right_i32"]
    if use_native:
        op = _pack_endpoint_records_i32_native_op()
        if op is not None:
            return op(owner_i32, left_i32, right_i32)
    owner = owner_i32.detach().cpu().to(dtype=torch.int64)
    if owner.numel():
        if int(owner.min().item()) < -1 or int(owner.max().item()) > 255:
            raise ValueError("packed endpoint records support owner ids in [-1, 255]")
    owner = torch.where(owner < 0, torch.zeros_like(owner), owner)
    left_code = _pack_cut_ids_i32(left_i32)
    right_code = _pack_cut_ids_i32(right_i32)
    packed = owner | (left_code << 8) | (right_code << 20)
    if packed.numel() and int(packed.max().item()) > 2_147_483_647:
        raise ValueError("packed endpoint record exceeded signed int32 range")
    return packed.to(dtype=torch.int32)


def _validate_packed_endpoint_record_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    expected_shape: torch.Size | tuple[int, ...],
    site_count: int | None = None,
    boundary_count: int | None = None,
) -> torch.Tensor:
    tensor = _validate_cpu_int32_vector_tensor(name, tensor, expected_shape=expected_shape)
    _validate_packed_endpoint_record_ranges(
        name,
        tensor,
        site_count=site_count,
        boundary_count=boundary_count,
    )
    return tensor


def _validate_cpu_int32_vector_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    expected_shape: torch.Size | tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor before MPS transfer")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be torch.int32")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match expected {tuple(expected_shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _validate_endpoint_record_components(
    name: str,
    *,
    owner_i32: torch.Tensor,
    left_i32: torch.Tensor,
    right_i32: torch.Tensor,
    site_count: int | None = None,
    boundary_count: int | None = None,
) -> dict[str, torch.Tensor]:
    owner_i32 = _validate_cpu_int32_vector_tensor(f"{name} owner_i32", owner_i32)
    left_i32 = _validate_cpu_int32_vector_tensor(f"{name} left_i32", left_i32)
    right_i32 = _validate_cpu_int32_vector_tensor(f"{name} right_i32", right_i32)
    if owner_i32.shape != left_i32.shape or owner_i32.shape != right_i32.shape:
        raise ValueError(f"{name} owner/left/right tensors must have matching shapes")
    _validate_endpoint_record_component_ranges(
        name,
        owner_i32=owner_i32,
        left_i32=left_i32,
        right_i32=right_i32,
        site_count=site_count,
        boundary_count=boundary_count,
    )
    return {
        "owner_i32": owner_i32,
        "left_i32": left_i32,
        "right_i32": right_i32,
    }


def _validate_endpoint_record_component_ranges(
    name: str,
    *,
    owner_i32: torch.Tensor,
    left_i32: torch.Tensor,
    right_i32: torch.Tensor,
    site_count: int | None = None,
    boundary_count: int | None = None,
) -> None:
    owner = owner_i32.detach().cpu().to(dtype=torch.int64)
    if owner.numel() and int(owner.min().item()) < -1:
        raise ValueError(f"{name} owner_i32 must be >= -1")
    if site_count is not None and owner.numel() and int(owner.max().item()) >= int(site_count):
        raise ValueError(f"{name} owner_i32 must be < site_count={int(site_count)}")
    _validate_endpoint_record_cut_range(f"{name} left_i32", left_i32, boundary_count=boundary_count)
    _validate_endpoint_record_cut_range(f"{name} right_i32", right_i32, boundary_count=boundary_count)


def _validate_endpoint_record_cut_range(
    name: str,
    cut_i32: torch.Tensor,
    *,
    boundary_count: int | None = None,
) -> None:
    cut = cut_i32.detach().cpu().to(dtype=torch.int64)
    if cut.numel() and int(cut.min().item()) < -2:
        raise ValueError(f"{name} must be >= -2")
    if boundary_count is not None and cut.numel() and int(cut.max().item()) >= int(boundary_count):
        raise ValueError(f"{name} must be < boundary_count={int(boundary_count)}")


def _unpack_endpoint_record_cut_code_i64(code_i64: torch.Tensor) -> torch.Tensor:
    return torch.where(
        code_i64 == 0,
        torch.full_like(code_i64, -1),
        torch.where(code_i64 == 1, torch.full_like(code_i64, -2), code_i64 - 2),
    )


def _validate_packed_endpoint_record_ranges(
    name: str,
    record_i32: torch.Tensor,
    *,
    site_count: int | None = None,
    boundary_count: int | None = None,
) -> None:
    if site_count is None and boundary_count is None:
        return
    packed = record_i32.detach().cpu().to(dtype=torch.int64)
    if packed.numel() and int(packed.min().item()) < 0:
        raise ValueError(f"{name} must contain nonnegative packed records")
    owner_i64 = packed & 255
    if site_count is not None and owner_i64.numel() and int(owner_i64.max().item()) >= int(site_count):
        raise ValueError(f"{name} owner code must be < site_count={int(site_count)}")
    if boundary_count is not None:
        left_i64 = _unpack_endpoint_record_cut_code_i64((packed >> 8) & 4095)
        right_i64 = _unpack_endpoint_record_cut_code_i64((packed >> 20) & 4095)
        if left_i64.numel() and int(left_i64.max().item()) >= int(boundary_count):
            raise ValueError(f"{name} left cut id must be < boundary_count={int(boundary_count)}")
        if right_i64.numel() and int(right_i64.max().item()) >= int(boundary_count):
            raise ValueError(f"{name} right cut id must be < boundary_count={int(boundary_count)}")


def _validate_monotonic_i32_offsets(name: str, offsets_i32: torch.Tensor, *, expected_final: int) -> None:
    if int(offsets_i32.numel()) < 1:
        raise ValueError(f"{name} must contain at least one offset")
    if int(offsets_i32[0].item()) != 0:
        raise ValueError(f"{name} must start at 0")
    if offsets_i32.numel() > 1 and bool((offsets_i32[1:] < offsets_i32[:-1]).any().item()):
        raise ValueError(f"{name} must be monotonic")
    if int(offsets_i32[-1].item()) != int(expected_final):
        raise ValueError(f"{name} final offset must equal {expected_final}")


def _validate_endpoint_delta_index_tables(delta: Any) -> dict[str, torch.Tensor]:
    base_offsets_i32 = _validate_cpu_int32_vector_tensor("base_offsets_i32", delta.base_offsets_i32)
    track_change_offsets_i32 = _validate_cpu_int32_vector_tensor(
        "track_change_offsets_i32",
        delta.track_change_offsets_i32,
    )
    change_frame_i32 = _validate_cpu_int32_vector_tensor("change_frame_i32", delta.change_frame_i32)
    change_offsets_i32 = _validate_cpu_int32_vector_tensor("change_offsets_i32", delta.change_offsets_i32)
    base_owner_i32 = _validate_cpu_int32_vector_tensor("base_owner_i32", delta.base_owner_i32)
    change_owner_i32 = _validate_cpu_int32_vector_tensor("change_owner_i32", delta.change_owner_i32)
    if int(base_offsets_i32.numel()) != int(track_change_offsets_i32.numel()):
        raise ValueError("base_offsets_i32 and track_change_offsets_i32 must have matching track counts")
    if int(change_offsets_i32.numel()) != int(change_frame_i32.numel()) + 1:
        raise ValueError("change_offsets_i32 length must equal change_frame_i32 length plus one")
    _validate_monotonic_i32_offsets(
        "base_offsets_i32",
        base_offsets_i32,
        expected_final=int(base_owner_i32.numel()),
    )
    _validate_monotonic_i32_offsets(
        "track_change_offsets_i32",
        track_change_offsets_i32,
        expected_final=int(change_frame_i32.numel()),
    )
    _validate_monotonic_i32_offsets(
        "change_offsets_i32",
        change_offsets_i32,
        expected_final=int(change_owner_i32.numel()),
    )
    return {
        "base_offsets_i32": base_offsets_i32,
        "track_change_offsets_i32": track_change_offsets_i32,
        "change_frame_i32": change_frame_i32,
        "change_offsets_i32": change_offsets_i32,
    }


def _native_owner_run_cutwalk_delta_op() -> Any | None:
    try:
        return torch.ops.world_foam_lane2_fused_slab_v0.gate4_owner_run_delta_replace_from_rays_cpu
    except (AttributeError, RuntimeError):
        return None


def _build_owner_run_delta_replace_native_cutwalk_tape(
    *,
    sites: tuple[Any, ...],
    boundaries: tuple[Any, ...],
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    site_rgba: torch.Tensor,
) -> Gate4EndpointDeltaReplaceTape:
    op = _native_owner_run_cutwalk_delta_op()
    if op is None:
        raise RuntimeError(
            "native owner-run cutwalk requested but gate4_owner_run_delta_replace_from_rays_cpu is unavailable"
        )
    boundary_f64 = torch.tensor(
        [[boundary.nx, boundary.ny, boundary.nz, boundary.nt, boundary.b] for boundary in boundaries],
        dtype=torch.float64,
    ).contiguous()
    site_f64 = torch.tensor(
        [[site.x, site.y, site.z, site.t, site.weight] for site in sites],
        dtype=torch.float64,
    ).contiguous()
    site_density_f32 = site_rgba.detach().cpu().to(dtype=torch.float32).contiguous()[:, 3].contiguous()
    rays = rays.detach().cpu().to(dtype=torch.float32).contiguous()
    frame_indices = frame_indices.detach().cpu().to(dtype=torch.long)
    sample_count, _height, _width, payload = rays.shape
    if payload != 6:
        raise ValueError(f"rays must have payload dimension 6, got {payload}")
    if sample_count % frame_count != 0:
        raise ValueError("sample count must be view_count * frame_count")
    result = op(
        boundary_f64,
        site_f64,
        site_density_f32,
        rays,
        frame_indices.contiguous(),
        int(frame_count),
        float(near),
        float(far),
        float(invalid_epsilon),
        float(transmittance_threshold),
        1.0e-6,
        1.0e-8,
    )
    return Gate4EndpointDeltaReplaceTape(
        base_offsets_i32=result[0].detach().cpu().to(dtype=torch.int32).contiguous(),
        base_owner_i32=result[1].detach().cpu().to(dtype=torch.int32).contiguous(),
        base_left_i32=result[2].detach().cpu().to(dtype=torch.int32).contiguous(),
        base_right_i32=result[3].detach().cpu().to(dtype=torch.int32).contiguous(),
        track_change_offsets_i32=result[4].detach().cpu().to(dtype=torch.int32).contiguous(),
        change_frame_i32=result[5].detach().cpu().to(dtype=torch.int32).contiguous(),
        change_offsets_i32=result[6].detach().cpu().to(dtype=torch.int32).contiguous(),
        change_owner_i32=result[7].detach().cpu().to(dtype=torch.int32).contiguous(),
        change_left_i32=result[8].detach().cpu().to(dtype=torch.int32).contiguous(),
        change_right_i32=result[9].detach().cpu().to(dtype=torch.int32).contiguous(),
    )


def _endpoint_records_to_i16(records_i32: torch.Tensor, name: str) -> torch.Tensor:
    records = records_i32.detach().cpu().to(dtype=torch.int64)
    if records.numel():
        if int(records.min().item()) < -32768 or int(records.max().item()) > 32767:
            raise ValueError(f"{name} must fit int16")
    return records.to(dtype=torch.int16)


def _pack_endpoint_records_i16x3(
    owner_i32: torch.Tensor,
    left_i32: torch.Tensor,
    right_i32: torch.Tensor,
    *,
    site_count: int | None = None,
    boundary_count: int | None = None,
) -> torch.Tensor:
    records = _validate_endpoint_record_components(
        "int16x3 endpoint record",
        owner_i32=owner_i32,
        left_i32=left_i32,
        right_i32=right_i32,
        site_count=site_count,
        boundary_count=boundary_count,
    )
    return torch.stack(
        (
            _endpoint_records_to_i16(records["owner_i32"], "owner_i16x3"),
            _endpoint_records_to_i16(records["left_i32"], "left_i16x3"),
            _endpoint_records_to_i16(records["right_i32"], "right_i16x3"),
        ),
        dim=1,
    ).reshape(-1)


def _pack_endpoint_records_i16cols(
    owner_i32: torch.Tensor,
    left_i32: torch.Tensor,
    right_i32: torch.Tensor,
    *,
    site_count: int | None = None,
    boundary_count: int | None = None,
) -> torch.Tensor:
    records = _validate_endpoint_record_components(
        "int16-column endpoint record",
        owner_i32=owner_i32,
        left_i32=left_i32,
        right_i32=right_i32,
        site_count=site_count,
        boundary_count=boundary_count,
    )
    return torch.cat(
        (
            _endpoint_records_to_i16(records["owner_i32"], "owner_i16cols"),
            _endpoint_records_to_i16(records["left_i32"], "left_i16cols"),
            _endpoint_records_to_i16(records["right_i32"], "right_i16cols"),
        ),
        dim=0,
    )


def _pack_endpoint_records_i16x4(
    owner_i32: torch.Tensor,
    left_i32: torch.Tensor,
    right_i32: torch.Tensor,
    *,
    site_count: int | None = None,
    boundary_count: int | None = None,
) -> torch.Tensor:
    records = _validate_endpoint_record_components(
        "int16x4 endpoint record",
        owner_i32=owner_i32,
        left_i32=left_i32,
        right_i32=right_i32,
        site_count=site_count,
        boundary_count=boundary_count,
    )
    return torch.stack(
        (
            _endpoint_records_to_i16(records["owner_i32"], "owner_i16x4"),
            _endpoint_records_to_i16(records["left_i32"], "left_i16x4"),
            _endpoint_records_to_i16(records["right_i32"], "right_i16x4"),
            torch.zeros_like(records["owner_i32"], dtype=torch.int16),
        ),
        dim=1,
    ).reshape(-1)


def _move_endpoint_record_block4_tape_to_mps(
    *,
    edit: Any,
    block_edit: Any,
    boundary_f32: torch.Tensor,
    rays_f32: torch.Tensor,
    frame_t_f32: torch.Tensor,
    coeff_f32: torch.Tensor | None = None,
    coeff_f16: torch.Tensor | None = None,
    packed_records: bool = False,
    i16_records: bool = False,
    i16x3_records: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = _move_endpoint_record_edit_tape_to_mps(
        edit=edit,
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
    )
    device = torch.device("mps")
    out.update(
        {
            "block_anchor_offsets_i32": block_edit.anchor_offsets_i32.to(device=device).contiguous(),
            "block_anchor_owner_i32": block_edit.anchor_owner_i32.to(device=device).contiguous(),
            "block_anchor_left_i32": block_edit.anchor_left_i32.to(device=device).contiguous(),
            "block_anchor_right_i32": block_edit.anchor_right_i32.to(device=device).contiguous(),
            "block_track_change_offsets_i32": block_edit.track_block_change_offsets_i32.to(device=device).contiguous(),
            "block_change_frame_i32": block_edit.change_frame_i32.to(device=device).contiguous(),
            "block_op_offsets_i32": block_edit.op_offsets_i32.to(device=device).contiguous(),
            "block_op_type_i32": block_edit.op_type_i32.to(device=device).contiguous(),
            "block_op_pos_i32": block_edit.op_pos_i32.to(device=device).contiguous(),
            "block_op_owner_i32": block_edit.op_owner_i32.to(device=device).contiguous(),
            "block_op_left_i32": block_edit.op_left_i32.to(device=device).contiguous(),
            "block_op_right_i32": block_edit.op_right_i32.to(device=device).contiguous(),
            "block_size": int(block_edit.block_size),
        }
    )
    if coeff_f32 is not None:
        out["block_coeff_f32"] = coeff_f32.to(device=device).contiguous()
        out["block_coeff_boundary_count"] = int(boundary_f32.shape[0])
    if coeff_f16 is not None:
        out["block_coeff_f16"] = coeff_f16.to(device=device, dtype=torch.float16).contiguous()
        out["block_coeff_boundary_count"] = int(boundary_f32.shape[0])
    if packed_records:
        out["block_anchor_record_i32"] = _pack_endpoint_records_i32(
            block_edit.anchor_owner_i32,
            block_edit.anchor_left_i32,
            block_edit.anchor_right_i32,
        ).to(device=device)
        out["block_op_record_i32"] = _pack_endpoint_records_i32(
            block_edit.op_owner_i32,
            block_edit.op_left_i32,
            block_edit.op_right_i32,
        ).to(device=device)
    if i16_records:
        out["block_anchor_owner_i16"] = _endpoint_records_to_i16(block_edit.anchor_owner_i32, "block_anchor_owner_i16").to(
            device=device
        ).contiguous()
        out["block_anchor_left_i16"] = _endpoint_records_to_i16(block_edit.anchor_left_i32, "block_anchor_left_i16").to(
            device=device
        ).contiguous()
        out["block_anchor_right_i16"] = _endpoint_records_to_i16(block_edit.anchor_right_i32, "block_anchor_right_i16").to(
            device=device
        ).contiguous()
        out["block_op_owner_i16"] = _endpoint_records_to_i16(block_edit.op_owner_i32, "block_op_owner_i16").to(
            device=device
        ).contiguous()
        out["block_op_left_i16"] = _endpoint_records_to_i16(block_edit.op_left_i32, "block_op_left_i16").to(
            device=device
        ).contiguous()
        out["block_op_right_i16"] = _endpoint_records_to_i16(block_edit.op_right_i32, "block_op_right_i16").to(
            device=device
        ).contiguous()
    if i16x3_records:
        out["block_anchor_record_i16x3"] = _pack_endpoint_records_i16x3(
            block_edit.anchor_owner_i32,
            block_edit.anchor_left_i32,
            block_edit.anchor_right_i32,
        ).to(device=device).contiguous()
        out["block_op_record_i16x3"] = _pack_endpoint_records_i16x3(
            block_edit.op_owner_i32,
            block_edit.op_left_i32,
            block_edit.op_right_i32,
        ).to(device=device).contiguous()
    return out


def _selected_tape_storage_bytes(
    *,
    tape_mode: str,
    selected: Any,
    endpoint_record_edit: Any | None,
    endpoint_record_delta_replace: Any | None = None,
    endpoint_record_block_edit: Any | None,
    coeff_f32: torch.Tensor | None,
    extra_storage_bytes: int = 0,
    factorized_coeff_storage_bytes: int = 0,
    record_bytes_override: int | None = None,
    include_delta_index_storage: bool = True,
) -> int:
    if tape_mode == "endpoint-record-delta-replace-coeff16-fused-mse":
        if endpoint_record_delta_replace is None:
            raise ValueError("endpoint-record-delta-replace-coeff16-fused-mse requires endpoint_record_delta_replace")
        if coeff_f32 is None:
            raise ValueError("endpoint-record-delta-replace-coeff16-fused-mse requires coeff_f32")
        return int(endpoint_record_delta_replace.storage_bytes) + int(coeff_f32.numel() * 2)
    if tape_mode in {
        "endpoint-record-delta-replace-coeff16-i16x3-fused-mse",
        DELTA_I16X3_FRAMEGROUP16_MODE,
        DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse",
        DELTA_PACKED_SCALAR_MODE,
        DELTA_PACKED_FRAMEGROUP16_MODE,
        DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
        DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
        DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
        OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        DELTA_AUTO_FRAMEGROUP16_MODE,
        "endpoint-record-delta-replace-coeff16-i16x4-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
    }:
        if endpoint_record_delta_replace is None:
            raise ValueError(f"{tape_mode} requires endpoint_record_delta_replace")
        if coeff_f32 is None and tape_mode not in OWNER_RUN_DELTA_PACKED_FACTORIZED_MODES:
            raise ValueError(f"{tape_mode} requires coeff_f32")
        record_bytes = (
            int(record_bytes_override)
            if record_bytes_override is not None
            else _delta_framegroup_record_bytes(tape_mode)
        )
        uses_i16_meta = tape_mode in OWNER_RUN_DELTA_PACKED_FACTORIZED_MODES
        storage_bytes = 0
        if include_delta_index_storage:
            if tape_mode == OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE:
                storage_bytes += int(
                    endpoint_record_delta_replace.base_offsets_i32.numel() * 4
                    + endpoint_record_delta_replace.track_change_offsets_i32.numel() * 4
                    + endpoint_record_delta_replace.change_offsets_i32.numel() * 4
                )
            else:
                index_element_size = 2 if uses_i16_meta else 4
                storage_tensors = [endpoint_record_delta_replace.base_offsets_i32]
                if tape_mode != OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE:
                    storage_tensors.extend(
                        (
                            endpoint_record_delta_replace.track_change_offsets_i32,
                        )
                    )
                    if tape_mode != OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE:
                        storage_tensors.append(endpoint_record_delta_replace.change_frame_i32)
                storage_tensors.append(endpoint_record_delta_replace.change_offsets_i32)
                storage_bytes += int(
                    sum(t.numel() * index_element_size for t in storage_tensors)
                )
        storage_bytes += int(endpoint_record_delta_replace.base_owner_i32.numel() * record_bytes)
        storage_bytes += int(endpoint_record_delta_replace.change_owner_i32.numel() * record_bytes)
        if tape_mode in OWNER_RUN_DELTA_PACKED_FACTORIZED_MODES:
            storage_bytes += int(factorized_coeff_storage_bytes)
        else:
            storage_bytes += int(coeff_f32.numel() * 2)
        return storage_bytes + int(extra_storage_bytes)
    if tape_mode in {
        "endpoint-record-edit-block4",
        "endpoint-record-edit-block-coeff",
        "endpoint-record-edit-block-coeff-rgb",
        "endpoint-record-edit-block-coeff-fused-mse",
        "endpoint-record-edit-block-coeff16",
        "endpoint-record-edit-block-coeff16-fused-mse",
        "endpoint-record-edit-block-coeff16-packed-fused-mse",
        "endpoint-record-edit-block-coeff16-i16-fused-mse",
        "endpoint-record-edit-block-coeff16-i16x3-fused-mse",
    }:
        if endpoint_record_block_edit is None:
            raise ValueError(f"{tape_mode} requires endpoint_record_block_edit")
        if tape_mode == "endpoint-record-edit-block-coeff16-packed-fused-mse":
            storage_tensors = (
                endpoint_record_block_edit.anchor_offsets_i32,
                endpoint_record_block_edit.track_block_change_offsets_i32,
                endpoint_record_block_edit.change_frame_i32,
                endpoint_record_block_edit.op_offsets_i32,
                endpoint_record_block_edit.op_type_i32,
                endpoint_record_block_edit.op_pos_i32,
            )
            storage_bytes = int(sum(t.numel() * t.element_size() for t in storage_tensors))
            storage_bytes += int(endpoint_record_block_edit.anchor_owner_i32.numel() * 4)
            storage_bytes += int(endpoint_record_block_edit.op_owner_i32.numel() * 4)
        elif tape_mode == "endpoint-record-edit-block-coeff16-i16-fused-mse":
            storage_tensors = (
                endpoint_record_block_edit.anchor_offsets_i32,
                endpoint_record_block_edit.track_block_change_offsets_i32,
                endpoint_record_block_edit.change_frame_i32,
                endpoint_record_block_edit.op_offsets_i32,
                endpoint_record_block_edit.op_type_i32,
                endpoint_record_block_edit.op_pos_i32,
            )
            storage_bytes = int(sum(t.numel() * t.element_size() for t in storage_tensors))
            storage_bytes += int(endpoint_record_block_edit.anchor_owner_i32.numel() * 6)
            storage_bytes += int(endpoint_record_block_edit.op_owner_i32.numel() * 6)
        elif tape_mode == "endpoint-record-edit-block-coeff16-i16x3-fused-mse":
            storage_tensors = (
                endpoint_record_block_edit.anchor_offsets_i32,
                endpoint_record_block_edit.track_block_change_offsets_i32,
                endpoint_record_block_edit.change_frame_i32,
                endpoint_record_block_edit.op_offsets_i32,
                endpoint_record_block_edit.op_type_i32,
                endpoint_record_block_edit.op_pos_i32,
            )
            storage_bytes = int(sum(t.numel() * t.element_size() for t in storage_tensors))
            storage_bytes += int(endpoint_record_block_edit.anchor_owner_i32.numel() * 6)
            storage_bytes += int(endpoint_record_block_edit.op_owner_i32.numel() * 6)
        else:
            storage_bytes = int(endpoint_record_block_edit.storage_bytes)
        if tape_mode in {
            "endpoint-record-edit-block-coeff",
            "endpoint-record-edit-block-coeff-rgb",
            "endpoint-record-edit-block-coeff-fused-mse",
        }:
            if coeff_f32 is None:
                raise ValueError(f"{tape_mode} requires coeff_f32")
            storage_bytes += int(coeff_f32.numel() * coeff_f32.element_size())
        elif tape_mode == "endpoint-record-edit-block-coeff16":
            if coeff_f32 is None:
                raise ValueError("endpoint-record-edit-block-coeff16 requires coeff_f32")
            storage_bytes += int(coeff_f32.numel() * 2)
        elif tape_mode == "endpoint-record-edit-block-coeff16-fused-mse":
            if coeff_f32 is None:
                raise ValueError("endpoint-record-edit-block-coeff16-fused-mse requires coeff_f32")
            storage_bytes += int(coeff_f32.numel() * 2)
        elif tape_mode == "endpoint-record-edit-block-coeff16-packed-fused-mse":
            if coeff_f32 is None:
                raise ValueError("endpoint-record-edit-block-coeff16-packed-fused-mse requires coeff_f32")
            storage_bytes += int(coeff_f32.numel() * 2)
        elif tape_mode == "endpoint-record-edit-block-coeff16-i16-fused-mse":
            if coeff_f32 is None:
                raise ValueError("endpoint-record-edit-block-coeff16-i16-fused-mse requires coeff_f32")
            storage_bytes += int(coeff_f32.numel() * 2)
        elif tape_mode == "endpoint-record-edit-block-coeff16-i16x3-fused-mse":
            if coeff_f32 is None:
                raise ValueError("endpoint-record-edit-block-coeff16-i16x3-fused-mse requires coeff_f32")
            storage_bytes += int(coeff_f32.numel() * 2)
        return storage_bytes
    if tape_mode in {
        "endpoint-record-edit",
        "endpoint-record-edit-fused-mse",
        "endpoint-record-edit-coeff16-fused-mse",
    }:
        if endpoint_record_edit is None:
            raise ValueError("endpoint-record-edit requires endpoint_record_edit")
        storage_bytes = int(endpoint_record_edit.storage_bytes)
        if tape_mode == "endpoint-record-edit-coeff16-fused-mse":
            if coeff_f32 is None:
                raise ValueError("endpoint-record-edit-coeff16-fused-mse requires coeff_f32")
            storage_bytes += int(coeff_f32.numel() * 2)
        return storage_bytes
    if tape_mode == OWNER_RUN_FUSED_MSE_NOMID_MODE:
        return int(
            selected.offsets_i32.numel() * selected.offsets_i32.element_size()
            + selected.owners_i32.numel() * selected.owners_i32.element_size()
            + selected.lengths_f32.numel() * selected.lengths_f32.element_size()
        )
    return int(selected.storage_bytes)


def _selected_coeff_storage_bytes(
    *,
    tape_mode: str,
    coeff_f32: torch.Tensor | None,
    factorized_coeff_storage_bytes: int = 0,
) -> int:
    if tape_mode in OWNER_RUN_DELTA_PACKED_FACTORIZED_MODES:
        return int(factorized_coeff_storage_bytes)
    if coeff_f32 is None:
        return 0
    if tape_mode in {
        "endpoint-record-edit-block-coeff",
        "endpoint-record-edit-block-coeff-rgb",
        "endpoint-record-edit-block-coeff-fused-mse",
    }:
        return int(coeff_f32.numel() * coeff_f32.element_size())
    if tape_mode in {
        "endpoint-record-edit-coeff16-fused-mse",
        "endpoint-record-delta-replace-coeff16-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x3-fused-mse",
        DELTA_I16X3_FRAMEGROUP16_MODE,
        DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse",
        DELTA_PACKED_SCALAR_MODE,
        DELTA_PACKED_FRAMEGROUP16_MODE,
        DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
        DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
        DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
        OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        DELTA_AUTO_FRAMEGROUP16_MODE,
        "endpoint-record-delta-replace-coeff16-i16x4-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
        "endpoint-record-edit-block-coeff16",
        "endpoint-record-edit-block-coeff16-fused-mse",
        "endpoint-record-edit-block-coeff16-packed-fused-mse",
        "endpoint-record-edit-block-coeff16-i16-fused-mse",
        "endpoint-record-edit-block-coeff16-i16x3-fused-mse",
    }:
        return int(coeff_f32.numel() * 2)
    return 0


def _selected_schema_storage_by_key(
    *,
    tape_mode: str,
    endpoint_record_delta_replace: Any | None,
    selected_storage_bytes: int,
    selected_coeff_storage_bytes: int,
    extra_storage_bytes: int = 0,
    record_bytes_override: int | None = None,
    include_delta_index_storage: bool = True,
) -> dict[str, int]:
    delta_modes = {
        "endpoint-record-delta-replace-coeff16-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x3-fused-mse",
        DELTA_I16X3_FRAMEGROUP16_MODE,
        DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse",
        DELTA_PACKED_SCALAR_MODE,
        DELTA_PACKED_FRAMEGROUP16_MODE,
        DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
        DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
        DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
        OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        DELTA_AUTO_FRAMEGROUP16_MODE,
        "endpoint-record-delta-replace-coeff16-i16x4-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
    }
    if tape_mode not in delta_modes or endpoint_record_delta_replace is None:
        return {
            "coeff_storage": int(selected_coeff_storage_bytes),
            "topology_storage": int(max(int(selected_storage_bytes) - int(selected_coeff_storage_bytes), 0)),
        }

    record_bytes = (
        int(record_bytes_override)
        if record_bytes_override is not None
        else _delta_framegroup_record_bytes(tape_mode)
    )
    by_key: dict[str, int] = {}
    uses_i16_meta = tape_mode in OWNER_RUN_DELTA_PACKED_FACTORIZED_MODES
    if include_delta_index_storage:
        if uses_i16_meta:
            if tape_mode == OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE:
                by_key["base_offsets_i32"] = _tensor_storage_bytes(endpoint_record_delta_replace.base_offsets_i32)
            else:
                by_key["base_offsets_i16"] = int(endpoint_record_delta_replace.base_offsets_i32.numel() * 2)
            if tape_mode != OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE:
                if tape_mode == OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE:
                    by_key["track_change_offsets_i32"] = _tensor_storage_bytes(
                        endpoint_record_delta_replace.track_change_offsets_i32
                    )
                else:
                    by_key["track_change_offsets_i16"] = int(
                        endpoint_record_delta_replace.track_change_offsets_i32.numel() * 2
                    )
                if tape_mode != OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE:
                    by_key["change_frame_i16"] = int(endpoint_record_delta_replace.change_frame_i32.numel() * 2)
            if tape_mode == OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE:
                by_key["change_offsets_i32"] = _tensor_storage_bytes(endpoint_record_delta_replace.change_offsets_i32)
            else:
                by_key["change_offsets_i16"] = int(endpoint_record_delta_replace.change_offsets_i32.numel() * 2)
        else:
            by_key["base_offsets_i32"] = _tensor_storage_bytes(endpoint_record_delta_replace.base_offsets_i32)
            by_key["track_change_offsets_i32"] = _tensor_storage_bytes(
                endpoint_record_delta_replace.track_change_offsets_i32
            )
            by_key["change_frame_i32"] = _tensor_storage_bytes(endpoint_record_delta_replace.change_frame_i32)
            by_key["change_offsets_i32"] = _tensor_storage_bytes(endpoint_record_delta_replace.change_offsets_i32)
    by_key["base_record_packed"] = int(endpoint_record_delta_replace.base_owner_i32.numel() * record_bytes)
    by_key["change_record_packed"] = int(endpoint_record_delta_replace.change_owner_i32.numel() * record_bytes)
    by_key[
        "factorized_coeff_f32"
        if tape_mode in OWNER_RUN_DELTA_PACKED_FACTORIZED_MODES
        else "delta_coeff_f16"
    ] = int(selected_coeff_storage_bytes)
    if extra_storage_bytes:
        by_key[
            "frame_select_i16"
            if tape_mode == OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE
            else "track_frame_mask_i32"
            if tape_mode == OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE
            else "extra_storage"
        ] = int(extra_storage_bytes)
    accounted = sum(by_key.values())
    if accounted != int(selected_storage_bytes):
        by_key["unattributed_storage"] = int(selected_storage_bytes) - accounted
    return by_key


def _int16_projection_info(tensor: torch.Tensor) -> dict[str, int | bool]:
    if int(tensor.numel()) == 0:
        return {"eligible": True, "min": 0, "max": 0, "bytes": 0, "projected_bytes": 0}
    cpu = tensor.detach().cpu()
    min_value = int(cpu.min().item())
    max_value = int(cpu.max().item())
    return {
        "eligible": bool(min_value >= -32768 and max_value <= 32767),
        "min": min_value,
        "max": max_value,
        "bytes": _tensor_storage_bytes(tensor),
        "projected_bytes": int(tensor.numel() * 2),
    }


def _to_mps_i16_meta_tensor(name: str, tensor: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    info = _int16_projection_info(tensor)
    if not bool(info["eligible"]):
        raise ValueError(
            f"{name} cannot use int16 metadata: min={info['min']} max={info['max']} outside int16 range"
        )
    return tensor.to(device=device, dtype=torch.int16).contiguous()


def _validated_delta_change_frame_lists(
    delta: Any,
    *,
    frame_count: int,
    selector_name: str,
) -> tuple[list[int], list[int]]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    track_offsets_i32 = _validate_cpu_int32_vector_tensor(
        "track_change_offsets_i32",
        delta.track_change_offsets_i32,
    )
    change_frame_i32 = _validate_cpu_int32_vector_tensor("change_frame_i32", delta.change_frame_i32)
    track_offsets = [int(value) for value in track_offsets_i32.tolist()]
    change_frames = [int(value) for value in change_frame_i32.tolist()]
    if not track_offsets:
        raise ValueError("track_change_offsets_i32 must contain at least one offset")
    if track_offsets[0] != 0:
        raise ValueError("track_change_offsets_i32 must start at 0")
    for current_offset, next_offset in zip(track_offsets, track_offsets[1:]):
        if next_offset < current_offset:
            raise ValueError("track_change_offsets_i32 must be monotonic")
    if track_offsets[-1] != len(change_frames):
        raise ValueError("track_change_offsets_i32 final offset must equal change_frame_i32 length")
    for track_id in range(len(track_offsets) - 1):
        previous_frame_id = 0
        for change_index in range(track_offsets[track_id], track_offsets[track_id + 1]):
            frame_id = change_frames[change_index]
            if frame_id <= 0 or frame_id >= frame_count:
                raise ValueError(
                    f"{selector_name} saw change frame {frame_id}, expected in [1, {frame_count})"
                )
            if frame_id <= previous_frame_id:
                raise ValueError(f"{selector_name} requires strictly ascending change frames per track")
            previous_frame_id = frame_id
    return track_offsets, change_frames


def _build_delta_frame_select_i16(delta: Any, *, frame_count: int) -> torch.Tensor:
    track_offsets, change_frames = _validated_delta_change_frame_lists(
        delta,
        frame_count=frame_count,
        selector_name="frame-select selector",
    )
    if len(change_frames) > 32767:
        raise ValueError("frame-select map stores sparse change indices as int16 and requires change count <= 32767")
    frame_slots = frame_count - 1
    out: list[int] = []
    for track_id in range(len(track_offsets) - 1):
        begin = track_offsets[track_id]
        end = track_offsets[track_id + 1]
        cursor = begin
        selected = -1
        for frame_id in range(1, frame_count):
            while cursor < end and change_frames[cursor] <= frame_id:
                if change_frames[cursor] >= 0:
                    selected = cursor
                cursor += 1
            out.append(selected)
    expected = (len(track_offsets) - 1) * frame_slots
    if len(out) != expected:
        raise ValueError(f"frame-select map has {len(out)} entries, expected {expected}")
    return torch.tensor(out, dtype=torch.int16)


def _build_delta_frame_bitmask_i32(delta: Any, *, frame_count: int) -> torch.Tensor:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frame_count > 32:
        raise ValueError("frame-bitmask selector stores frame ids in int32 bits and requires frame_count <= 32")
    track_offsets, change_frames = _validated_delta_change_frame_lists(
        delta,
        frame_count=frame_count,
        selector_name="frame-bitmask selector",
    )
    out: list[int] = []
    for track_id in range(len(track_offsets) - 1):
        begin = track_offsets[track_id]
        end = track_offsets[track_id + 1]
        mask = 0
        for change_index in range(begin, end):
            frame_id = change_frames[change_index]
            bit = 1 << frame_id
            mask |= bit
        if mask >= (1 << 31):
            mask -= 1 << 32
        out.append(mask)
    return torch.tensor(out, dtype=torch.int32)


def _selected_schema_i16_meta_projection(
    *,
    endpoint_record_delta_replace: Any | None,
    selected_schema_storage_by_key: dict[str, int],
    selected_storage_bytes: int,
) -> dict[str, Any]:
    if endpoint_record_delta_replace is None:
        return {
            "eligible": False,
            "storage_bytes": int(selected_storage_bytes),
            "savings_bytes": 0,
            "by_key": dict(selected_schema_storage_by_key),
            "fields": {},
        }

    field_map = (
        ("base_offsets_i32", "base_offsets_i16", endpoint_record_delta_replace.base_offsets_i32),
        (
            "track_change_offsets_i32",
            "track_change_offsets_i16",
            endpoint_record_delta_replace.track_change_offsets_i32,
        ),
        ("change_frame_i32", "change_frame_i16", endpoint_record_delta_replace.change_frame_i32),
        ("change_offsets_i32", "change_offsets_i16", endpoint_record_delta_replace.change_offsets_i32),
    )
    fields = {old_key: _int16_projection_info(tensor) for old_key, _new_key, tensor in field_map}
    by_key = dict(selected_schema_storage_by_key)
    for old_key, new_key, _tensor in field_map:
        if old_key not in by_key:
            continue
        info = fields[old_key]
        if bool(info["eligible"]):
            by_key.pop(old_key, None)
            by_key[new_key] = int(info["projected_bytes"])
    projected_storage_bytes = int(sum(by_key.values()))
    return {
        "eligible": all(bool(info["eligible"]) for info in fields.values()),
        "storage_bytes": projected_storage_bytes,
        "savings_bytes": int(selected_storage_bytes) - projected_storage_bytes,
        "by_key": by_key,
        "fields": fields,
    }


def _tensor_storage_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _selected_device_tensor_storage_breakdown(
    selected_device: dict[str, Any],
    *,
    device_type: str | None = "mps",
) -> dict[str, Any]:
    by_key: dict[str, int] = {}
    coeff_bytes = 0
    total_bytes = 0
    for key, value in sorted(selected_device.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if device_type is not None and value.device.type != device_type:
            continue
        byte_count = _tensor_storage_bytes(value)
        by_key[key] = byte_count
        total_bytes += byte_count
        if "coeff" in key:
            coeff_bytes += byte_count
    return {
        "total_bytes": int(total_bytes),
        "coeff_bytes": int(coeff_bytes),
        "noncoeff_bytes": int(max(total_bytes - coeff_bytes, 0)),
        "by_key": by_key,
    }


def _with_counts(tape: SegmentTape, counts_i32: torch.Tensor) -> SegmentTape:
    return SegmentTape(
        owners_i32=tape.owners_i32,
        lengths_f32=tape.lengths_f32,
        mids_f32=tape.mids_f32,
        counts_i32=counts_i32.to(dtype=torch.int32).reshape(tape.track_count, tape.frame_count).contiguous(),
        active_counts_i32=counts_i32.to(dtype=torch.int32).reshape(tape.track_count, tape.frame_count).contiguous(),
        frame_t_f32=tape.frame_t_f32,
        track_count=tape.track_count,
        frame_count=tape.frame_count,
        max_segments=tape.max_segments,
    )


def _render_owner_run_rgb(
    *,
    tape_device: dict[str, torch.Tensor],
    site_rgba: torch.Tensor,
    op_config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    view_count: int,
    height: int,
    width: int,
    autograd_vjp_mode: str | None = None,
) -> torch.Tensor:
    if "affine_candidate_row_offsets_i32" in tape_device:
        if autograd_vjp_mode is not None:
            raise ValueError("Gate4 affine candidate CSR currently supports manual-vjp train/eval only")
        if "affine_candidate_depth_coeff_f16" in tape_device:
            rgb, _alpha, _depth = fused_slab_affine_coeff16_realray_rgba_depth_replay(
                tape_device["affine_row_index_i32"],
                tape_device["affine_candidate_row_offsets_i32"],
                tape_device["affine_candidate_depth_coeff_f16"],
                tape_device["affine_sites_f32"],
                site_rgba,
                tape_device["affine_ray_f32"],
                tape_device["affine_frame_t_f32"],
                op_config,
                time_slab_count=int(tape_device["affine_time_slab_count"]),
                row_count=int(tape_device["affine_row_count"]),
            )
        else:
            rgb, _alpha, _depth = fused_slab_affine_num32_den16_realray_rgba_depth_replay(
                tape_device["affine_row_index_i32"],
                tape_device["affine_candidate_row_offsets_i32"],
                tape_device["affine_candidate_depth_num_f32"],
                tape_device["affine_candidate_depth_den_f16"],
                tape_device["affine_sites_f32"],
                site_rgba,
                tape_device["affine_ray_f32"],
                tape_device["affine_frame_t_f32"],
                op_config,
                time_slab_count=int(tape_device["affine_time_slab_count"]),
                row_count=int(tape_device["affine_row_count"]),
            )
    elif "block_anchor_offsets_i32" in tape_device and "block_coeff_f16" in tape_device:
        if autograd_vjp_mode is not None:
            raise ValueError("endpoint-record-edit-block-coeff16 currently supports manual-vjp train/eval only")
        coeff16_args = (
            tape_device["block_coeff_f16"],
            tape_device["frame_t_f32"],
            tape_device["block_anchor_offsets_i32"],
            tape_device["block_anchor_owner_i32"],
            tape_device["block_anchor_left_i32"],
            tape_device["block_anchor_right_i32"],
            tape_device["block_track_change_offsets_i32"],
            tape_device["block_change_frame_i32"],
            tape_device["block_op_offsets_i32"],
            tape_device["block_op_type_i32"],
            tape_device["block_op_pos_i32"],
            tape_device["block_op_owner_i32"],
            tape_device["block_op_left_i32"],
            tape_device["block_op_right_i32"],
        )
        rgb, _alpha, _depth = endpoint_record_edit_block_coeff16_rgba_depth_replay(
            *coeff16_args,
            site_rgba,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["block_coeff_boundary_count"]),
            block_size=int(tape_device["block_size"]),
        )
    elif "block_anchor_offsets_i32" in tape_device and "block_coeff_f32" in tape_device:
        coeff_args = (
            tape_device["block_coeff_f32"],
            tape_device["frame_t_f32"],
            tape_device["block_anchor_offsets_i32"],
            tape_device["block_anchor_owner_i32"],
            tape_device["block_anchor_left_i32"],
            tape_device["block_anchor_right_i32"],
            tape_device["block_track_change_offsets_i32"],
            tape_device["block_change_frame_i32"],
            tape_device["block_op_offsets_i32"],
            tape_device["block_op_type_i32"],
            tape_device["block_op_pos_i32"],
            tape_device["block_op_owner_i32"],
            tape_device["block_op_left_i32"],
            tape_device["block_op_right_i32"],
        )
        if bool(tape_device.get("block_coeff_rgb_only", False)):
            if autograd_vjp_mode is None:
                rgb = endpoint_record_edit_block_coeff_rgb_replay(
                    *coeff_args,
                    site_rgba,
                    op_config,
                    track_count=track_count,
                    frame_count=frame_count,
                    boundary_count=int(tape_device["block_coeff_boundary_count"]),
                    block_size=int(tape_device["block_size"]),
                )
            else:
                rgb = endpoint_record_edit_block_coeff_rgb_autograd(
                    coeff_args[0],
                    tape_device["boundary_f32"],
                    tape_device["rays_f32"],
                    *coeff_args[1:],
                    tape_device["base_offsets_i32"],
                    tape_device["base_owner_i32"],
                    tape_device["base_left_i32"],
                    tape_device["base_right_i32"],
                    tape_device["track_change_offsets_i32"],
                    tape_device["change_frame_i32"],
                    tape_device["op_offsets_i32"],
                    tape_device["op_type_i32"],
                    tape_device["op_pos_i32"],
                    tape_device["op_owner_i32"],
                    tape_device["op_left_i32"],
                    tape_device["op_right_i32"],
                    site_rgba,
                    op_config,
                    track_count=track_count,
                    frame_count=frame_count,
                    boundary_count=int(tape_device["block_coeff_boundary_count"]),
                    block_size=int(tape_device["block_size"]),
                )
        elif autograd_vjp_mode is None:
            rgb, _alpha, _depth = endpoint_record_edit_block_coeff_rgba_depth_replay(
                *coeff_args,
                site_rgba,
                op_config,
                track_count=track_count,
                frame_count=frame_count,
                boundary_count=int(tape_device["block_coeff_boundary_count"]),
                block_size=int(tape_device["block_size"]),
            )
        else:
            rgb, _alpha, _depth = endpoint_record_edit_block_coeff_rgba_depth_autograd(
                coeff_args[0],
                tape_device["boundary_f32"],
                tape_device["rays_f32"],
                *coeff_args[1:],
                tape_device["base_offsets_i32"],
                tape_device["base_owner_i32"],
                tape_device["base_left_i32"],
                tape_device["base_right_i32"],
                tape_device["track_change_offsets_i32"],
                tape_device["change_frame_i32"],
                tape_device["op_offsets_i32"],
                tape_device["op_type_i32"],
                tape_device["op_pos_i32"],
                tape_device["op_owner_i32"],
                tape_device["op_left_i32"],
                tape_device["op_right_i32"],
                site_rgba,
                op_config,
                track_count=track_count,
                frame_count=frame_count,
                boundary_count=int(tape_device["block_coeff_boundary_count"]),
                block_size=int(tape_device["block_size"]),
            )
    elif "block_anchor_offsets_i32" in tape_device:
        block_args = (
            tape_device["boundary_f32"],
            tape_device["rays_f32"],
            tape_device["frame_t_f32"],
            tape_device["block_anchor_offsets_i32"],
            tape_device["block_anchor_owner_i32"],
            tape_device["block_anchor_left_i32"],
            tape_device["block_anchor_right_i32"],
            tape_device["block_track_change_offsets_i32"],
            tape_device["block_change_frame_i32"],
            tape_device["block_op_offsets_i32"],
            tape_device["block_op_type_i32"],
            tape_device["block_op_pos_i32"],
            tape_device["block_op_owner_i32"],
            tape_device["block_op_left_i32"],
            tape_device["block_op_right_i32"],
        )
        if autograd_vjp_mode is None:
            rgb, _alpha, _depth = endpoint_record_edit_block4_rgba_depth_replay(
                *block_args,
                site_rgba,
                op_config,
                track_count=track_count,
                frame_count=frame_count,
                block_size=int(tape_device["block_size"]),
            )
        else:
            rgb, _alpha, _depth = endpoint_record_edit_block4_rgba_depth_autograd(
                *block_args,
                tape_device["base_offsets_i32"],
                tape_device["base_owner_i32"],
                tape_device["base_left_i32"],
                tape_device["base_right_i32"],
                tape_device["track_change_offsets_i32"],
                tape_device["change_frame_i32"],
                tape_device["op_offsets_i32"],
                tape_device["op_type_i32"],
                tape_device["op_pos_i32"],
                tape_device["op_owner_i32"],
                tape_device["op_left_i32"],
                tape_device["op_right_i32"],
                site_rgba,
                op_config,
                track_count=track_count,
                frame_count=frame_count,
                block_size=int(tape_device["block_size"]),
            )
    elif "change_offsets_i32" in tape_device and "op_type_i32" not in tape_device:
        if autograd_vjp_mode is not None:
            raise ValueError("endpoint-record-delta-replace currently supports manual-vjp train/eval only")
        rgb, _alpha, _depth = endpoint_record_delta_replace_rgba_depth_replay(
            tape_device["boundary_f32"],
            tape_device["rays_f32"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i32"],
            tape_device["base_owner_i32"],
            tape_device["base_left_i32"],
            tape_device["base_right_i32"],
            tape_device["track_change_offsets_i32"],
            tape_device["change_frame_i32"],
            tape_device["change_offsets_i32"],
            tape_device["change_owner_i32"],
            tape_device["change_left_i32"],
            tape_device["change_right_i32"],
            site_rgba,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
        )
    elif "op_type_i32" in tape_device:
        replay = (
            endpoint_record_edit_rgba_depth_replay
            if autograd_vjp_mode is None
            else endpoint_record_edit_rgba_depth_autograd
        )
        rgb, _alpha, _depth = replay(
            tape_device["boundary_f32"],
            tape_device["rays_f32"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i32"],
            tape_device["base_owner_i32"],
            tape_device["base_left_i32"],
            tape_device["base_right_i32"],
            tape_device["track_change_offsets_i32"],
            tape_device["change_frame_i32"],
            tape_device["op_offsets_i32"],
            tape_device["op_type_i32"],
            tape_device["op_pos_i32"],
            tape_device["op_owner_i32"],
            tape_device["op_left_i32"],
            tape_device["op_right_i32"],
            site_rgba,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
        )
    elif "starts_f32" in tape_device:
        replay = endpoint_run_rgba_depth_replay if autograd_vjp_mode is None else endpoint_run_rgba_depth_autograd
        rgb, _alpha, _depth = replay(
            tape_device["offsets_i32"],
            tape_device["owners_i32"],
            tape_device["starts_f32"],
            tape_device["ends_f32"],
            site_rgba,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
        )
    else:
        replay = segment_tape_rgba_depth_replay if autograd_vjp_mode is None else segment_tape_rgba_depth_autograd
        kwargs = {} if autograd_vjp_mode is None else {"vjp_mode": autograd_vjp_mode}
        rgb, _alpha, _depth = replay(
            tape_device["offsets_i32"],
            tape_device["owners_i32"],
            tape_device["lengths_f32"],
            tape_device["mids_f32"],
            site_rgba,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            **kwargs,
        )
    return _image_rgb_from_track_major(
        rgb.reshape(track_count, frame_count, 3),
        view_count=view_count,
        frame_count=frame_count,
        height=height,
        width=width,
    )


def _owner_run_vjp_rgb_only(
    *,
    tape_device: dict[str, torch.Tensor],
    site_rgba: torch.Tensor,
    grad_rgb_image: torch.Tensor,
    op_config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    view_count: int,
    height: int,
    width: int,
) -> torch.Tensor:
    grad_rgb = _track_major_grad_from_image(
        grad_rgb_image,
        view_count=view_count,
        frame_count=frame_count,
        height=height,
        width=width,
    )
    if "block_anchor_offsets_i32" in tape_device and "block_coeff_f16" in tape_device:
        return endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only(
            tape_device["block_coeff_f16"],
            tape_device["frame_t_f32"],
            tape_device["block_anchor_offsets_i32"],
            tape_device["block_anchor_owner_i32"],
            tape_device["block_anchor_left_i32"],
            tape_device["block_anchor_right_i32"],
            tape_device["block_track_change_offsets_i32"],
            tape_device["block_change_frame_i32"],
            tape_device["block_op_offsets_i32"],
            tape_device["block_op_type_i32"],
            tape_device["block_op_pos_i32"],
            tape_device["block_op_owner_i32"],
            tape_device["block_op_left_i32"],
            tape_device["block_op_right_i32"],
            site_rgba,
            grad_rgb,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["block_coeff_boundary_count"]),
            block_size=int(tape_device["block_size"]),
        )
    if "block_anchor_offsets_i32" in tape_device and "block_coeff_f32" in tape_device:
        return endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
            tape_device["block_coeff_f32"],
            tape_device["frame_t_f32"],
            tape_device["block_anchor_offsets_i32"],
            tape_device["block_anchor_owner_i32"],
            tape_device["block_anchor_left_i32"],
            tape_device["block_anchor_right_i32"],
            tape_device["block_track_change_offsets_i32"],
            tape_device["block_change_frame_i32"],
            tape_device["block_op_offsets_i32"],
            tape_device["block_op_type_i32"],
            tape_device["block_op_pos_i32"],
            tape_device["block_op_owner_i32"],
            tape_device["block_op_left_i32"],
            tape_device["block_op_right_i32"],
            site_rgba,
            grad_rgb,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["block_coeff_boundary_count"]),
            block_size=int(tape_device["block_size"]),
        )
    if "block_anchor_offsets_i32" in tape_device:
        return endpoint_record_edit_block4_vjp_direct_atomic_rgb_only(
            tape_device["boundary_f32"],
            tape_device["rays_f32"],
            tape_device["frame_t_f32"],
            tape_device["block_anchor_offsets_i32"],
            tape_device["block_anchor_owner_i32"],
            tape_device["block_anchor_left_i32"],
            tape_device["block_anchor_right_i32"],
            tape_device["block_track_change_offsets_i32"],
            tape_device["block_change_frame_i32"],
            tape_device["block_op_offsets_i32"],
            tape_device["block_op_type_i32"],
            tape_device["block_op_pos_i32"],
            tape_device["block_op_owner_i32"],
            tape_device["block_op_left_i32"],
            tape_device["block_op_right_i32"],
            site_rgba,
            grad_rgb,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            block_size=int(tape_device["block_size"]),
        )
    if "op_type_i32" in tape_device:
        return endpoint_record_edit_vjp_direct_atomic_rgb_only(
            tape_device["boundary_f32"],
            tape_device["rays_f32"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i32"],
            tape_device["base_owner_i32"],
            tape_device["base_left_i32"],
            tape_device["base_right_i32"],
            tape_device["track_change_offsets_i32"],
            tape_device["change_frame_i32"],
            tape_device["op_offsets_i32"],
            tape_device["op_type_i32"],
            tape_device["op_pos_i32"],
            tape_device["op_owner_i32"],
            tape_device["op_left_i32"],
            tape_device["op_right_i32"],
            site_rgba,
            grad_rgb,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
        )
    grad_alpha = torch.zeros((track_count, frame_count), dtype=torch.float32, device=site_rgba.device)
    grad_depth = torch.zeros_like(grad_alpha)
    if "starts_f32" in tape_device:
        return endpoint_run_vjp_direct_atomic_grad_only(
            tape_device["offsets_i32"],
            tape_device["owners_i32"],
            tape_device["starts_f32"],
            tape_device["ends_f32"],
            site_rgba,
            grad_rgb,
            grad_alpha,
            grad_depth,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
        )
    return segment_tape_vjp_direct_atomic_grad_only(
        tape_device["offsets_i32"],
        tape_device["owners_i32"],
        tape_device["lengths_f32"],
        tape_device["mids_f32"],
        site_rgba,
        grad_rgb,
        grad_alpha,
        grad_depth,
        op_config,
        track_count=track_count,
        frame_count=frame_count,
    )


def _endpoint_run_fused_mse_loss_vjp(
    *,
    tape_device: dict[str, torch.Tensor],
    site_rgba: torch.Tensor,
    target_rgb_track: torch.Tensor,
    op_config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return endpoint_run_mse_vjp_direct_atomic_rgb_only(
        tape_device["offsets_i32"],
        tape_device["owners_i32"],
        tape_device["starts_f32"],
        tape_device["ends_f32"],
        site_rgba,
        target_rgb_track,
        op_config,
        track_count=track_count,
        frame_count=frame_count,
    )


def _segment_tape_fused_mse_loss_vjp(
    *,
    tape_device: dict[str, torch.Tensor],
    site_rgba: torch.Tensor,
    target_rgb_track: torch.Tensor,
    op_config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "mids_f32" not in tape_device:
        return segment_tape_nomids_mse_vjp_direct_atomic_rgb_only(
            tape_device["offsets_i32"],
            tape_device["owners_i32"],
            tape_device["lengths_f32"],
            site_rgba,
            target_rgb_track,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
        )
    return segment_tape_mse_vjp_direct_atomic_rgb_only(
        tape_device["offsets_i32"],
        tape_device["owners_i32"],
        tape_device["lengths_f32"],
        tape_device["mids_f32"],
        site_rgba,
        target_rgb_track,
        op_config,
        track_count=track_count,
        frame_count=frame_count,
    )


def _affine_candidate_fused_mse_loss_vjp(
    *,
    tape_device: dict[str, torch.Tensor],
    site_rgba: torch.Tensor,
    target_rgb_track: torch.Tensor,
    op_config: RealRayReplayConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "affine_candidate_row_offsets_i32" not in tape_device:
        raise ValueError("Gate4 affine candidate fused MSE path requires affine candidate CSR tape")
    trackmse = bool(tape_device.get("affine_candidate_trackmse_fused_mse", False))
    cap224 = bool(tape_device.get("affine_candidate_cap224_fused_mse", False))
    densitymask = bool(tape_device.get("affine_candidate_densitymask_fused_mse", False))
    sample_reduce = bool(tape_device.get("affine_candidate_sample_reduce_fused_mse", False))
    sortnet = bool(tape_device.get("affine_candidate_sortnet_fused_mse", False))
    framegroup16_cached = bool(tape_device.get("affine_candidate_framegroup16_cached_fused_mse", False))
    sitecache = bool(tape_device.get("affine_candidate_sitecache_fused_mse", False))
    ownerupdate = bool(tape_device.get("affine_candidate_ownerupdate_fused_mse", False))
    ownerupdate_i16 = bool(tape_device.get("affine_candidate_ownerupdate_i16_fused_mse", False))
    ownerkeep = bool(tape_device.get("affine_candidate_ownerkeep_fused_mse", False))
    ownerkeep_i16 = bool(tape_device.get("affine_candidate_ownerkeep_i16_fused_mse", False))
    if "affine_candidate_depth_coeff_f16" in tape_device:
        if ownerkeep_i16:
            return fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only(
                tape_device["affine_row_index_i32"],
                tape_device["affine_candidate_row_offsets_i32"],
                tape_device["affine_candidate_boundary_ids_i16"],
                tape_device["affine_candidate_depth_coeff_f16"],
                tape_device["affine_boundary_site_pairs_i16"],
                tape_device["affine_sites_f32"],
                site_rgba,
                tape_device["affine_ray_f32"],
                tape_device["affine_frame_t_f32"],
                target_rgb_track,
                op_config,
                time_slab_count=int(tape_device["affine_time_slab_count"]),
                row_count=int(tape_device["affine_row_count"]),
            )
        if ownerupdate_i16:
            return fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only(
                tape_device["affine_row_index_i32"],
                tape_device["affine_candidate_row_offsets_i32"],
                tape_device["affine_candidate_boundary_ids_i16"],
                tape_device["affine_candidate_depth_coeff_f16"],
                tape_device["affine_boundary_site_pairs_i16"],
                tape_device["affine_sites_f32"],
                site_rgba,
                tape_device["affine_ray_f32"],
                tape_device["affine_frame_t_f32"],
                target_rgb_track,
                op_config,
                time_slab_count=int(tape_device["affine_time_slab_count"]),
                row_count=int(tape_device["affine_row_count"]),
            )
        if ownerupdate:
            return fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(
                tape_device["affine_row_index_i32"],
                tape_device["affine_candidate_row_offsets_i32"],
                tape_device["affine_candidate_boundary_ids_i32"],
                tape_device["affine_candidate_depth_coeff_f16"],
                tape_device["affine_boundary_site_pairs_i32"],
                tape_device["affine_sites_f32"],
                site_rgba,
                tape_device["affine_ray_f32"],
                tape_device["affine_frame_t_f32"],
                target_rgb_track,
                op_config,
                time_slab_count=int(tape_device["affine_time_slab_count"]),
                row_count=int(tape_device["affine_row_count"]),
            )
        if ownerkeep:
            return fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only(
                tape_device["affine_row_index_i32"],
                tape_device["affine_candidate_row_offsets_i32"],
                tape_device["affine_candidate_boundary_ids_i32"],
                tape_device["affine_candidate_depth_coeff_f16"],
                tape_device["affine_boundary_site_pairs_i32"],
                tape_device["affine_sites_f32"],
                site_rgba,
                tape_device["affine_ray_f32"],
                tape_device["affine_frame_t_f32"],
                target_rgb_track,
                op_config,
                time_slab_count=int(tape_device["affine_time_slab_count"]),
                row_count=int(tape_device["affine_row_count"]),
            )
        if cap224:
            return fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only(
                tape_device["affine_row_index_i32"],
                tape_device["affine_candidate_row_offsets_i32"],
                tape_device["affine_candidate_depth_coeff_f16"],
                tape_device["affine_sites_f32"],
                site_rgba,
                tape_device["affine_ray_f32"],
                tape_device["affine_frame_t_f32"],
                target_rgb_track,
                op_config,
                time_slab_count=int(tape_device["affine_time_slab_count"]),
                row_count=int(tape_device["affine_row_count"]),
            )
        if densitymask:
            return fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only(
                tape_device["affine_row_index_i32"],
                tape_device["affine_candidate_row_offsets_i32"],
                tape_device["affine_candidate_depth_coeff_f16"],
                tape_device["affine_sites_f32"],
                site_rgba,
                tape_device["affine_ray_f32"],
                tape_device["affine_frame_t_f32"],
                target_rgb_track,
                op_config,
                time_slab_count=int(tape_device["affine_time_slab_count"]),
                row_count=int(tape_device["affine_row_count"]),
            )
        if sample_reduce:
            return fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only(
                tape_device["affine_row_index_i32"],
                tape_device["affine_candidate_row_offsets_i32"],
                tape_device["affine_candidate_depth_coeff_f16"],
                tape_device["affine_sites_f32"],
                site_rgba,
                tape_device["affine_ray_f32"],
                tape_device["affine_frame_t_f32"],
                target_rgb_track,
                op_config,
                time_slab_count=int(tape_device["affine_time_slab_count"]),
                row_count=int(tape_device["affine_row_count"]),
            )
        if sortnet:
            return fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only(
                tape_device["affine_row_index_i32"],
                tape_device["affine_candidate_row_offsets_i32"],
                tape_device["affine_candidate_depth_coeff_f16"],
                tape_device["affine_sites_f32"],
                site_rgba,
                tape_device["affine_ray_f32"],
                tape_device["affine_frame_t_f32"],
                target_rgb_track,
                op_config,
                time_slab_count=int(tape_device["affine_time_slab_count"]),
                row_count=int(tape_device["affine_row_count"]),
            )
        if framegroup16_cached:
            return fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only(
                tape_device["affine_row_index_i32"],
                tape_device["affine_candidate_row_offsets_i32"],
                tape_device["affine_candidate_depth_coeff_f16"],
                tape_device["affine_sites_f32"],
                site_rgba,
                tape_device["affine_ray_f32"],
                tape_device["affine_frame_t_f32"],
                target_rgb_track,
                op_config,
                time_slab_count=int(tape_device["affine_time_slab_count"]),
                row_count=int(tape_device["affine_row_count"]),
            )
        if sitecache:
            return fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only(
                tape_device["affine_row_index_i32"],
                tape_device["affine_candidate_row_offsets_i32"],
                tape_device["affine_candidate_depth_coeff_f16"],
                tape_device["affine_sites_f32"],
                site_rgba,
                tape_device["affine_ray_f32"],
                tape_device["affine_frame_t_f32"],
                target_rgb_track,
                op_config,
                time_slab_count=int(tape_device["affine_time_slab_count"]),
                row_count=int(tape_device["affine_row_count"]),
            )
        op = (
            fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only
            if trackmse
            else fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only
        )
        return op(
            tape_device["affine_row_index_i32"],
            tape_device["affine_candidate_row_offsets_i32"],
            tape_device["affine_candidate_depth_coeff_f16"],
            tape_device["affine_sites_f32"],
            site_rgba,
            tape_device["affine_ray_f32"],
            tape_device["affine_frame_t_f32"],
            target_rgb_track,
            op_config,
            time_slab_count=int(tape_device["affine_time_slab_count"]),
            row_count=int(tape_device["affine_row_count"]),
        )
    op = (
        fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only
        if trackmse
        else fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only
    )
    return op(
        tape_device["affine_row_index_i32"],
        tape_device["affine_candidate_row_offsets_i32"],
        tape_device["affine_candidate_depth_num_f32"],
        tape_device["affine_candidate_depth_den_f16"],
        tape_device["affine_sites_f32"],
        site_rgba,
        tape_device["affine_ray_f32"],
        tape_device["affine_frame_t_f32"],
        target_rgb_track,
        op_config,
        time_slab_count=int(tape_device["affine_time_slab_count"]),
        row_count=int(tape_device["affine_row_count"]),
    )


def _edit_fused_mse_loss_vjp(
    *,
    tape_device: dict[str, torch.Tensor],
    site_rgba: torch.Tensor,
    target_rgb_track: torch.Tensor,
    op_config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "op_type_i32" not in tape_device or "block_anchor_offsets_i32" in tape_device:
        raise ValueError("raw edit fused MSE path requires endpoint-record-edit tape")
    if "edit_coeff_f16" in tape_device:
        if "edit_config_i32" in tape_device and "edit_config_f32" in tape_device:
            op_name = "endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only"
            ops = torch.ops.world_foam_lane2_fused_slab_v0
            if not hasattr(ops, op_name):
                raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
            return getattr(ops, op_name)(
                tape_device["edit_coeff_f16"],
                tape_device["frame_t_f32"],
                tape_device["base_offsets_i32"],
                tape_device["base_owner_i32"],
                tape_device["base_left_i32"],
                tape_device["base_right_i32"],
                tape_device["track_change_offsets_i32"],
                tape_device["change_frame_i32"],
                tape_device["op_offsets_i32"],
                tape_device["op_type_i32"],
                tape_device["op_pos_i32"],
                tape_device["op_owner_i32"],
                tape_device["op_left_i32"],
                tape_device["op_right_i32"],
                site_rgba,
                target_rgb_track,
                tape_device["edit_config_i32"],
                tape_device["edit_config_f32"],
            )
        return endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only(
            tape_device["edit_coeff_f16"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i32"],
            tape_device["base_owner_i32"],
            tape_device["base_left_i32"],
            tape_device["base_right_i32"],
            tape_device["track_change_offsets_i32"],
            tape_device["change_frame_i32"],
            tape_device["op_offsets_i32"],
            tape_device["op_type_i32"],
            tape_device["op_pos_i32"],
            tape_device["op_owner_i32"],
            tape_device["op_left_i32"],
            tape_device["op_right_i32"],
            site_rgba,
            target_rgb_track,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["edit_coeff_boundary_count"]),
        )
    if "edit_config_i32" in tape_device and "edit_config_f32" in tape_device:
        op_name = "endpoint_record_edit_mse_vjp_direct_atomic_rgb_only"
        ops = torch.ops.world_foam_lane2_fused_slab_v0
        if not hasattr(ops, op_name):
            raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
        return getattr(ops, op_name)(
            tape_device["boundary_f32"],
            tape_device["rays_f32"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i32"],
            tape_device["base_owner_i32"],
            tape_device["base_left_i32"],
            tape_device["base_right_i32"],
            tape_device["track_change_offsets_i32"],
            tape_device["change_frame_i32"],
            tape_device["op_offsets_i32"],
            tape_device["op_type_i32"],
            tape_device["op_pos_i32"],
            tape_device["op_owner_i32"],
            tape_device["op_left_i32"],
            tape_device["op_right_i32"],
            site_rgba,
            target_rgb_track,
            tape_device["edit_config_i32"],
            tape_device["edit_config_f32"],
        )
    return endpoint_record_edit_mse_vjp_direct_atomic_rgb_only(
        tape_device["boundary_f32"],
        tape_device["rays_f32"],
        tape_device["frame_t_f32"],
        tape_device["base_offsets_i32"],
        tape_device["base_owner_i32"],
        tape_device["base_left_i32"],
        tape_device["base_right_i32"],
        tape_device["track_change_offsets_i32"],
        tape_device["change_frame_i32"],
        tape_device["op_offsets_i32"],
        tape_device["op_type_i32"],
        tape_device["op_pos_i32"],
        tape_device["op_owner_i32"],
        tape_device["op_left_i32"],
        tape_device["op_right_i32"],
        site_rgba,
        target_rgb_track,
        op_config,
        track_count=track_count,
        frame_count=frame_count,
    )


_PACKED_DIRECT_CONFIG_TENSOR_KEYS = (
    "boundary_f32",
    "delta_coeff_f16",
    "track_ray_coeff_f32",
    "frame_t_f32",
    "base_offsets_i32",
    "base_offsets_i16",
    "base_owner_i32",
    "base_left_i32",
    "base_right_i32",
    "delta_base_record_i32",
    "delta_base_record_i16x4",
    "delta_base_record_i16cols",
    "delta_base_record_i16x3",
    "track_change_offsets_i32",
    "track_change_offsets_i16",
    "track_chunk_change_offsets_i16",
    "track_chunk_owner_offsets_i32",
    "track_chunk_owner_i16",
    "change_frame_i32",
    "change_frame_i16",
    "change_offsets_i32",
    "change_offsets_i16",
    "frame_change_index_i16",
    "track_frame_mask_i32",
    "row_begin_i32",
    "row_len_source_i16",
    "change_owner_i32",
    "change_left_i32",
    "change_right_i32",
    "delta_change_record_i32",
    "delta_change_record_i16x4",
    "delta_change_record_i16cols",
    "delta_change_record_i16x3",
    "delta_config_i32",
    "delta_config_f32",
)
_PACKED_DIRECT_CONFIG_SELECTOR_KEYS = (
    "delta_packed_scalar_fused_mse",
    "delta_packed_framegroup16_fused_mse",
    "delta_packed_framegroup16_materialized_fused_mse",
    "delta_packed_framegroup16_recompute_fused_mse",
    "delta_packed_framegroup16_factorized_recompute_fused_mse",
    "delta_packed_frameselect_factorized_recompute_fused_mse",
    "delta_packed_framebitmask_factorized_recompute_fused_mse",
    "delta_packed_framegroup16_smallrun16_fused_mse",
    "delta_packed_framegroup16_launch_only_fused_mse",
    "delta_packed_framegroup16_unchecked_launch_only_fused_mse",
    "delta_packed_framegroup16_reduce32_launch_only_fused_mse",
    "delta_packed_framegroup16_rowselect32_launch_only_fused_mse",
    "delta_packed_framegroup16_rowdesc_launch_only_fused_mse",
    "delta_packed_framegroup16_rowdesc32_launch_only_fused_mse",
    "delta_i16x4_framegroup16_fused_mse",
    "delta_i16cols_framegroup16_fused_mse",
    "delta_i16x3_framegroup16_fused_mse",
    "delta_i16x3_framegroup16_materialized_fused_mse",
    "delta_i16x3_framegroup16_ownerreduce_fused_mse",
    "delta_i16x3_framegroup64_fused_mse",
)
_PACKED_DIRECT_CONFIG_PACKED_PRIMARY_SELECTOR_KEYS = (
    "delta_packed_scalar_fused_mse",
    "delta_packed_framegroup16_fused_mse",
    "delta_packed_framegroup16_materialized_fused_mse",
    "delta_packed_framegroup16_recompute_fused_mse",
    "delta_packed_framegroup16_smallrun16_fused_mse",
)
_PACKED_DIRECT_CONFIG_FACTORIZED_PRIMARY_SELECTOR_KEYS = (
    "delta_packed_framegroup16_factorized_recompute_fused_mse",
    "delta_packed_frameselect_factorized_recompute_fused_mse",
    "delta_packed_framebitmask_factorized_recompute_fused_mse",
)
_PACKED_DIRECT_CONFIG_I16_PRIMARY_SELECTOR_KEYS = (
    "delta_i16x4_framegroup16_fused_mse",
    "delta_i16cols_framegroup16_fused_mse",
    "delta_i16x3_framegroup16_fused_mse",
    "delta_i16x3_framegroup16_materialized_fused_mse",
    "delta_i16x3_framegroup16_ownerreduce_fused_mse",
    "delta_i16x3_framegroup64_fused_mse",
)
_PACKED_DIRECT_CONFIG_LAUNCH_MODIFIER_SELECTOR_KEYS = (
    "delta_packed_framegroup16_launch_only_fused_mse",
    "delta_packed_framegroup16_unchecked_launch_only_fused_mse",
    "delta_packed_framegroup16_reduce32_launch_only_fused_mse",
    "delta_packed_framegroup16_rowselect32_launch_only_fused_mse",
    "delta_packed_framegroup16_rowdesc_launch_only_fused_mse",
    "delta_packed_framegroup16_rowdesc32_launch_only_fused_mse",
)
_PACKED_DIRECT_CONFIG_SCALAR_KEYS = (
    "delta_coeff_boundary_count",
    "delta_launch_boundary_count",
    "delta_launch_track_count",
    "delta_launch_frame_count",
    "delta_launch_site_count",
    "delta_launch_base_record_count",
    "delta_launch_change_count",
    "delta_launch_change_record_count",
)
_PACKED_DIRECT_CONFIG_LAUNCH_COUNT_SCALAR_KEYS = (
    "delta_launch_boundary_count",
    "delta_launch_track_count",
    "delta_launch_frame_count",
    "delta_launch_site_count",
    "delta_launch_base_record_count",
    "delta_launch_change_count",
    "delta_launch_change_record_count",
)
_PACKED_DIRECT_CONFIG_REQUIRED_SCALAR_KEYS = (
    "delta_coeff_boundary_count",
) + _PACKED_DIRECT_CONFIG_LAUNCH_COUNT_SCALAR_KEYS


def _present_direct_config_keys(tape_device: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(key for key in keys if key in tape_device)


def _validate_packed_direct_config_selector_contract(tape_device: dict[str, Any]) -> None:
    packed_primary = _present_direct_config_keys(tape_device, _PACKED_DIRECT_CONFIG_PACKED_PRIMARY_SELECTOR_KEYS)
    factorized_primary = _present_direct_config_keys(
        tape_device,
        _PACKED_DIRECT_CONFIG_FACTORIZED_PRIMARY_SELECTOR_KEYS,
    )
    i16_primary = _present_direct_config_keys(tape_device, _PACKED_DIRECT_CONFIG_I16_PRIMARY_SELECTOR_KEYS)
    primary_selectors = packed_primary + factorized_primary + i16_primary
    if len(primary_selectors) > 1:
        raise ValueError(f"direct-config path has conflicting primary selectors: {list(primary_selectors)}")

    launch_modifiers = _present_direct_config_keys(tape_device, _PACKED_DIRECT_CONFIG_LAUNCH_MODIFIER_SELECTOR_KEYS)
    if launch_modifiers:
        if not packed_primary or packed_primary == ("delta_packed_scalar_fused_mse",):
            raise ValueError(
                "direct-config launch-only modifiers require a non-scalar packed framegroup selector"
            )
        if "delta_packed_framegroup16_launch_only_fused_mse" not in tape_device:
            raise ValueError(
                "direct-config launch-only modifiers require delta_packed_framegroup16_launch_only_fused_mse"
            )

    if (
        "delta_packed_framegroup16_rowdesc32_launch_only_fused_mse" in tape_device
        and "delta_packed_framegroup16_rowdesc_launch_only_fused_mse" not in tape_device
    ):
        raise ValueError("rowdesc32 launch-only selector requires rowdesc launch-only selector")

    row_selector_modifiers = _present_direct_config_keys(
        tape_device,
        (
            "delta_packed_framegroup16_reduce32_launch_only_fused_mse",
            "delta_packed_framegroup16_rowselect32_launch_only_fused_mse",
            "delta_packed_framegroup16_rowdesc_launch_only_fused_mse",
        ),
    )
    if len(row_selector_modifiers) > 1:
        raise ValueError(
            f"direct-config launch-only row selector modifiers are mutually exclusive: {list(row_selector_modifiers)}"
        )


def _tensor_identity_version_marker(key: str, value: object) -> tuple[object, ...]:
    if isinstance(value, torch.Tensor):
        return (key, id(value), int(value._version))
    return (key, type(value).__name__, id(value))


def _packed_direct_config_expected_dtype(key: str) -> torch.dtype:
    if key.endswith("_f32"):
        return torch.float32
    if key.endswith("_f16"):
        return torch.float16
    if key.endswith("_i32"):
        return torch.int32
    if "_i16" in key:
        return torch.int16
    raise ValueError(f"no expected dtype registered for direct-config tensor {key}")


def _validate_packed_direct_config_tensor_storage(
    *,
    tape_device: dict[str, Any],
    runtime_device: torch.device,
) -> None:
    for key in _PACKED_DIRECT_CONFIG_TENSOR_KEYS:
        if key not in tape_device:
            continue
        value = tape_device[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{key} must be a tensor for direct-config launch, got {type(value).__name__}")
        expected_dtype = _packed_direct_config_expected_dtype(key)
        if value.dtype != expected_dtype:
            raise ValueError(f"{key} must be {expected_dtype}, got {value.dtype}")
        if value.device != runtime_device:
            raise ValueError(f"{key} must be on {runtime_device}, got {value.device}")
        if not value.is_contiguous():
            raise ValueError(f"{key} must be contiguous")


def _direct_config_tensor_shape(tape_device: dict[str, Any], key: str) -> tuple[int, ...] | None:
    value = tape_device.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    return tuple(int(dim) for dim in value.shape)


def _require_direct_config_shape(tape_device: dict[str, Any], key: str, expected: tuple[int, ...]) -> None:
    actual = _direct_config_tensor_shape(tape_device, key)
    if actual is not None and actual != expected:
        raise ValueError(f"{key} must have shape {expected}, got {actual}")


def _require_direct_config_1d(tape_device: dict[str, Any], key: str) -> None:
    actual = _direct_config_tensor_shape(tape_device, key)
    if actual is not None and len(actual) != 1:
        raise ValueError(f"{key} must be a 1D tensor, got shape {actual}")


def _require_direct_config_flat_record_multiple(tape_device: dict[str, Any], key: str, multiple: int) -> None:
    actual = _direct_config_tensor_shape(tape_device, key)
    if actual is None:
        return
    if len(actual) != 1:
        raise ValueError(f"{key} must be a 1D tensor, got shape {actual}")
    if actual[0] % int(multiple) != 0:
        raise ValueError(f"{key} length must be a multiple of {int(multiple)}, got {actual[0]}")


def _require_direct_config_same_1d_len(tape_device: dict[str, Any], keys: tuple[str, ...]) -> None:
    lengths = {
        key: _direct_config_tensor_shape(tape_device, key)[0]
        for key in keys
        if _direct_config_tensor_shape(tape_device, key) is not None
    }
    if len(lengths) != len(keys):
        return
    expected = next(iter(lengths.values()))
    for key, actual in lengths.items():
        if actual != expected:
            raise ValueError(f"{key} length must match {keys[0]} length {expected}, got {actual}")


def _validate_packed_direct_config_tensor_layouts(
    *,
    tape_device: dict[str, Any],
    track_count: int,
    frame_count: int,
) -> None:
    for key in (
        "base_offsets_i32",
        "base_offsets_i16",
        "base_owner_i32",
        "base_left_i32",
        "base_right_i32",
        "delta_base_record_i32",
        "track_change_offsets_i32",
        "track_change_offsets_i16",
        "track_chunk_change_offsets_i16",
        "track_chunk_owner_offsets_i32",
        "track_chunk_owner_i16",
        "change_frame_i32",
        "change_frame_i16",
        "change_offsets_i32",
        "change_offsets_i16",
        "frame_change_index_i16",
        "track_frame_mask_i32",
        "row_begin_i32",
        "row_len_source_i16",
        "change_owner_i32",
        "change_left_i32",
        "change_right_i32",
        "delta_change_record_i32",
    ):
        _require_direct_config_1d(tape_device, key)

    boundary_shape = _direct_config_tensor_shape(tape_device, "boundary_f32")
    if boundary_shape is not None and (len(boundary_shape) != 2 or boundary_shape[1] != 5):
        raise ValueError(f"boundary_f32 must have shape [boundary_count,5], got {boundary_shape}")
    coeff_shape = _direct_config_tensor_shape(tape_device, "delta_coeff_f16")
    if coeff_shape is not None and (len(coeff_shape) != 2 or coeff_shape[1] != 4):
        raise ValueError(f"delta_coeff_f16 must have shape [row_count,4], got {coeff_shape}")

    _require_direct_config_shape(tape_device, "track_ray_coeff_f32", (int(track_count), 12))
    _require_direct_config_shape(tape_device, "frame_t_f32", (int(frame_count),))
    for key in ("base_offsets_i32", "base_offsets_i16", "track_change_offsets_i32", "track_change_offsets_i16"):
        _require_direct_config_shape(tape_device, key, (int(track_count) + 1,))
    _require_direct_config_shape(tape_device, "track_frame_mask_i32", (int(track_count),))
    _require_direct_config_shape(
        tape_device,
        "frame_change_index_i16",
        (int(track_count) * max(int(frame_count) - 1, 0),),
    )
    _require_direct_config_shape(tape_device, "row_begin_i32", (int(track_count) * int(frame_count),))
    _require_direct_config_shape(tape_device, "row_len_source_i16", (int(track_count) * int(frame_count),))

    change_frame_i32_shape = _direct_config_tensor_shape(tape_device, "change_frame_i32")
    if change_frame_i32_shape is not None:
        _require_direct_config_shape(tape_device, "change_offsets_i32", (change_frame_i32_shape[0] + 1,))
    change_frame_i16_shape = _direct_config_tensor_shape(tape_device, "change_frame_i16")
    if change_frame_i16_shape is not None:
        _require_direct_config_shape(tape_device, "change_offsets_i16", (change_frame_i16_shape[0] + 1,))

    _require_direct_config_flat_record_multiple(tape_device, "delta_base_record_i16x4", 4)
    _require_direct_config_flat_record_multiple(tape_device, "delta_change_record_i16x4", 4)
    for key in (
        "delta_base_record_i16x3",
        "delta_change_record_i16x3",
        "delta_base_record_i16cols",
        "delta_change_record_i16cols",
    ):
        _require_direct_config_flat_record_multiple(tape_device, key, 3)

    _require_direct_config_same_1d_len(tape_device, ("base_owner_i32", "base_left_i32", "base_right_i32"))
    _require_direct_config_same_1d_len(tape_device, ("change_owner_i32", "change_left_i32", "change_right_i32"))


def _direct_config_int_scalar(tape_device: dict[str, Any], key: str) -> int | None:
    if key not in tape_device:
        return None
    value = tape_device[key]
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{key} must be a Python integer scalar, got {type(value).__name__}")
    return int(value)


def _first_present_tensor_len(tape_device: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, int] | None:
    for key in keys:
        value = tape_device.get(key)
        if isinstance(value, torch.Tensor):
            return key, int(value.shape[0])
    return None


def _first_present_change_count_source(tape_device: dict[str, Any]) -> tuple[str, int] | None:
    frame_source = _first_present_tensor_len(tape_device, ("change_frame_i32", "change_frame_i16"))
    if frame_source is not None:
        return frame_source
    offset_source = _first_present_tensor_len(tape_device, ("change_offsets_i32", "change_offsets_i16"))
    if offset_source is None:
        return None
    source_key, offset_count = offset_source
    return source_key, max(offset_count - 1, 0)


def _direct_config_scalar_marker(key: str, value: object) -> tuple[object, ...]:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return _tensor_identity_version_marker(key, value)
    return (key, int(value))


def _validate_packed_direct_config_scalars(
    *,
    tape_device: dict[str, Any],
    site_count: int,
    track_count: int,
    frame_count: int,
) -> None:
    scalar_values = {
        key: _direct_config_int_scalar(tape_device, key)
        for key in _PACKED_DIRECT_CONFIG_SCALAR_KEYS
        if key in tape_device
    }
    i32_packed_selectors = _present_direct_config_keys(
        tape_device,
        _PACKED_DIRECT_CONFIG_PACKED_PRIMARY_SELECTOR_KEYS + _PACKED_DIRECT_CONFIG_FACTORIZED_PRIMARY_SELECTOR_KEYS,
    )
    if i32_packed_selectors:
        missing = sorted(key for key in _PACKED_DIRECT_CONFIG_REQUIRED_SCALAR_KEYS if key not in tape_device)
        if missing:
            raise ValueError(f"i32 packed direct-config path missing scalar contract keys: {missing}")
    for key, value in scalar_values.items():
        if value is None:
            continue
        if key in {
            "delta_coeff_boundary_count",
            "delta_launch_boundary_count",
            "delta_launch_track_count",
            "delta_launch_frame_count",
            "delta_launch_site_count",
        }:
            if value <= 0:
                raise ValueError(f"{key} must be positive, got {value}")
        elif value < 0:
            raise ValueError(f"{key} must be non-negative, got {value}")

    boundary_count = scalar_values.get("delta_coeff_boundary_count")
    if boundary_count is not None and "boundary_f32" in tape_device:
        expected = int(tape_device["boundary_f32"].shape[0])
        if boundary_count != expected:
            raise ValueError(f"delta_coeff_boundary_count must match boundary_f32 rows {expected}, got {boundary_count}")
    launch_boundary_count = scalar_values.get("delta_launch_boundary_count")
    if launch_boundary_count is not None:
        expected = boundary_count if boundary_count is not None else int(tape_device["boundary_f32"].shape[0])
        if launch_boundary_count != expected:
            raise ValueError(f"delta_launch_boundary_count must match boundary count {expected}, got {launch_boundary_count}")

    launch_track_count = scalar_values.get("delta_launch_track_count")
    if launch_track_count is not None and launch_track_count != int(track_count):
        raise ValueError(f"delta_launch_track_count must match runtime track_count {int(track_count)}, got {launch_track_count}")
    launch_frame_count = scalar_values.get("delta_launch_frame_count")
    if launch_frame_count is not None and launch_frame_count != int(frame_count):
        raise ValueError(f"delta_launch_frame_count must match runtime frame_count {int(frame_count)}, got {launch_frame_count}")
    launch_site_count = scalar_values.get("delta_launch_site_count")
    if launch_site_count is not None and launch_site_count != int(site_count):
        raise ValueError(f"delta_launch_site_count must match runtime site_count {int(site_count)}, got {launch_site_count}")

    base_record_count = scalar_values.get("delta_launch_base_record_count")
    base_record_source = _first_present_tensor_len(
        tape_device,
        ("delta_base_record_i32", "delta_base_record_i16x4", "delta_base_record_i16cols", "delta_base_record_i16x3"),
    )
    if base_record_count is not None and base_record_source is not None:
        source_key, expected = base_record_source
        if base_record_count != expected:
            raise ValueError(f"delta_launch_base_record_count must match {source_key} rows {expected}, got {base_record_count}")

    change_count = scalar_values.get("delta_launch_change_count")
    change_count_source = _first_present_change_count_source(tape_device)
    if change_count is not None and change_count_source is not None:
        source_key, expected = change_count_source
        if change_count != expected:
            raise ValueError(f"delta_launch_change_count must match {source_key} count {expected}, got {change_count}")

    change_record_count = scalar_values.get("delta_launch_change_record_count")
    change_record_source = _first_present_tensor_len(
        tape_device,
        ("delta_change_record_i32", "delta_change_record_i16x4", "delta_change_record_i16cols", "delta_change_record_i16x3"),
    )
    if change_record_count is not None and change_record_source is not None:
        source_key, expected = change_record_source
        if change_record_count != expected:
            raise ValueError(f"delta_launch_change_record_count must match {source_key} rows {expected}, got {change_record_count}")

    if "delta_config_i32" in tape_device:
        expected_config_len = 8 if "delta_i16x3_framegroup16_ownerreduce_fused_mse" in tape_device else 7
        actual_config_shape = tuple(tape_device["delta_config_i32"].shape)
        if actual_config_shape != (expected_config_len,):
            raise ValueError(f"delta_config_i32 must have shape ({expected_config_len},), got {actual_config_shape}")
    if "delta_config_f32" in tape_device and tuple(tape_device["delta_config_f32"].shape) != (4,):
        raise ValueError(f"delta_config_f32 must have shape (4,), got {tuple(tape_device['delta_config_f32'].shape)}")


def _packed_endpoint_direct_config_validation_marker(
    *,
    tape_device: dict[str, Any],
    site_count: int,
    track_count: int,
    frame_count: int,
) -> tuple[object, ...]:
    tensor_markers = tuple(
        _tensor_identity_version_marker(key, tape_device[key])
        for key in _PACKED_DIRECT_CONFIG_TENSOR_KEYS
        if key in tape_device
    )
    selector_markers = tuple(key for key in _PACKED_DIRECT_CONFIG_SELECTOR_KEYS if key in tape_device)
    scalar_markers = tuple(
        _direct_config_scalar_marker(key, tape_device[key])
        for key in _PACKED_DIRECT_CONFIG_SCALAR_KEYS
        if key in tape_device
    )
    return (
        "delta_direct_config_v8",
        tensor_markers,
        selector_markers,
        scalar_markers,
        int(site_count),
        int(track_count),
        int(frame_count),
    )


def _require_current_packed_endpoint_records_validated(
    *,
    tape_device: dict[str, Any],
    site_count: int,
    track_count: int,
    frame_count: int,
    runtime_device: torch.device,
) -> None:
    expected_marker = _packed_endpoint_direct_config_validation_marker(
        tape_device=tape_device,
        site_count=int(site_count),
        track_count=int(track_count),
        frame_count=int(frame_count),
    )
    if tape_device.get("delta_packed_records_validated") != expected_marker:
        raise ValueError(
            "delta direct-config path requires a prevalidated launch contract for the current tensors"
        )
    _validate_packed_direct_config_selector_contract(tape_device)
    _validate_packed_direct_config_tensor_storage(
        tape_device=tape_device,
        runtime_device=runtime_device,
    )
    _validate_packed_direct_config_tensor_layouts(
        tape_device=tape_device,
        track_count=track_count,
        frame_count=frame_count,
    )
    _validate_packed_direct_config_scalars(
        tape_device=tape_device,
        site_count=site_count,
        track_count=track_count,
        frame_count=frame_count,
    )


def _validate_track_major_fused_mse_runtime_tensors(
    *,
    site_rgba: torch.Tensor,
    target_rgb_track: torch.Tensor,
    track_count: int,
    frame_count: int,
) -> None:
    if site_rgba.ndim != 2 or int(site_rgba.shape[1]) != 4:
        raise ValueError(f"site_rgba must have shape [site_count,4], got {tuple(site_rgba.shape)}")
    expected_target_shape = (int(track_count), int(frame_count), 3)
    if tuple(target_rgb_track.shape) != expected_target_shape:
        raise ValueError(
            f"target_rgb_track must have shape {expected_target_shape}, got {tuple(target_rgb_track.shape)}"
        )
    if site_rgba.dtype != torch.float32:
        raise ValueError(f"site_rgba must be float32, got {site_rgba.dtype}")
    if target_rgb_track.dtype != torch.float32:
        raise ValueError(f"target_rgb_track must be float32, got {target_rgb_track.dtype}")
    if site_rgba.device != target_rgb_track.device:
        raise ValueError(
            f"site_rgba and target_rgb_track must be on the same device, got {site_rgba.device} and {target_rgb_track.device}"
        )
    if not site_rgba.is_contiguous():
        raise ValueError("site_rgba must be contiguous")
    if not target_rgb_track.is_contiguous():
        raise ValueError("target_rgb_track must be contiguous")


def _delta_replace_coeff16_fused_mse_loss_vjp(
    *,
    tape_device: dict[str, torch.Tensor],
    site_rgba: torch.Tensor,
    target_rgb_track: torch.Tensor,
    op_config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_track_major_fused_mse_runtime_tensors(
        site_rgba=site_rgba,
        target_rgb_track=target_rgb_track,
        track_count=track_count,
        frame_count=frame_count,
    )
    if "delta_packed_framebitmask_factorized_recompute_fused_mse" in tape_device:
        required = {
            "boundary_f32",
            "track_ray_coeff_f32",
            "frame_t_f32",
            "base_offsets_i32",
            "delta_base_record_i32",
            "track_change_offsets_i32",
            "track_frame_mask_i32",
            "change_offsets_i32",
            "delta_change_record_i32",
            "delta_coeff_boundary_count",
        }
        missing = sorted(required.difference(tape_device))
        if missing:
            raise ValueError(f"frame-bitmask factorized packed delta fused MSE path missing keys: {missing}")
        _require_current_packed_endpoint_records_validated(
            tape_device=tape_device,
            site_count=int(site_rgba.shape[0]),
            track_count=int(track_count),
            frame_count=int(frame_count),
            runtime_device=site_rgba.device,
        )
        return endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only(
            tape_device["boundary_f32"],
            tape_device["track_ray_coeff_f32"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i32"],
            tape_device["delta_base_record_i32"],
            tape_device["track_change_offsets_i32"],
            tape_device["track_frame_mask_i32"],
            tape_device["change_offsets_i32"],
            tape_device["delta_change_record_i32"],
            site_rgba,
            target_rgb_track,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["delta_coeff_boundary_count"]),
        )
    if "delta_packed_frameselect_factorized_recompute_fused_mse" in tape_device:
        required = {
            "boundary_f32",
            "track_ray_coeff_f32",
            "frame_t_f32",
            "base_offsets_i16",
            "delta_base_record_i32",
            "frame_change_index_i16",
            "change_offsets_i16",
            "delta_change_record_i32",
            "delta_coeff_boundary_count",
        }
        missing = sorted(required.difference(tape_device))
        if missing:
            raise ValueError(f"frame-select factorized packed delta fused MSE path missing keys: {missing}")
        _require_current_packed_endpoint_records_validated(
            tape_device=tape_device,
            site_count=int(site_rgba.shape[0]),
            track_count=int(track_count),
            frame_count=int(frame_count),
            runtime_device=site_rgba.device,
        )
        return endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only(
            tape_device["boundary_f32"],
            tape_device["track_ray_coeff_f32"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i16"],
            tape_device["delta_base_record_i32"],
            tape_device["frame_change_index_i16"],
            tape_device["change_offsets_i16"],
            tape_device["delta_change_record_i32"],
            site_rgba,
            target_rgb_track,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["delta_coeff_boundary_count"]),
        )
    if "delta_packed_framegroup16_factorized_recompute_fused_mse" in tape_device:
        required = {
            "boundary_f32",
            "track_ray_coeff_f32",
            "frame_t_f32",
            "base_offsets_i16",
            "delta_base_record_i32",
            "track_change_offsets_i16",
            "track_chunk_change_offsets_i16",
            "change_frame_i16",
            "change_offsets_i16",
            "delta_change_record_i32",
            "delta_coeff_boundary_count",
        }
        missing = sorted(required.difference(tape_device))
        if missing:
            raise ValueError(f"factorized packed delta fused MSE path missing keys: {missing}")
        _require_current_packed_endpoint_records_validated(
            tape_device=tape_device,
            site_count=int(site_rgba.shape[0]),
            track_count=int(track_count),
            frame_count=int(frame_count),
            runtime_device=site_rgba.device,
        )
        return endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(
            tape_device["boundary_f32"],
            tape_device["track_ray_coeff_f32"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i16"],
            tape_device["delta_base_record_i32"],
            tape_device["track_change_offsets_i16"],
            tape_device["track_chunk_change_offsets_i16"],
            tape_device["change_frame_i16"],
            tape_device["change_offsets_i16"],
            tape_device["delta_change_record_i32"],
            site_rgba,
            target_rgb_track,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["delta_coeff_boundary_count"]),
        )
    if "delta_coeff_f16" not in tape_device:
        raise ValueError("delta-replace coeff16 fused MSE path requires delta-replace tape")
    if "delta_base_record_i32" in tape_device and "delta_change_record_i32" in tape_device:
        use_packed_scalar = "delta_packed_scalar_fused_mse" in tape_device
        use_packed_materialized = "delta_packed_framegroup16_materialized_fused_mse" in tape_device
        use_packed_recompute = "delta_packed_framegroup16_recompute_fused_mse" in tape_device
        use_packed_smallrun16 = "delta_packed_framegroup16_smallrun16_fused_mse" in tape_device
        use_packed_launch_only = "delta_packed_framegroup16_launch_only_fused_mse" in tape_device
        use_packed_unchecked_launch_only = "delta_packed_framegroup16_unchecked_launch_only_fused_mse" in tape_device
        use_packed_reduce32_launch_only = "delta_packed_framegroup16_reduce32_launch_only_fused_mse" in tape_device
        use_packed_rowselect32_launch_only = "delta_packed_framegroup16_rowselect32_launch_only_fused_mse" in tape_device
        use_packed_rowdesc_launch_only = "delta_packed_framegroup16_rowdesc_launch_only_fused_mse" in tape_device
        use_packed_rowdesc32_launch_only = "delta_packed_framegroup16_rowdesc32_launch_only_fused_mse" in tape_device
        if "delta_config_i32" in tape_device and "delta_config_f32" in tape_device:
            _require_current_packed_endpoint_records_validated(
                tape_device=tape_device,
                site_count=int(site_rgba.shape[0]),
                track_count=int(track_count),
                frame_count=int(frame_count),
                runtime_device=site_rgba.device,
            )
            if use_packed_rowselect32_launch_only:
                if use_packed_unchecked_launch_only:
                    op_name = (
                        "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only"
                    )
                else:
                    op_name = (
                        "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_launch_only_mse_vjp_direct_atomic_rgb_only"
                    )
            elif use_packed_reduce32_launch_only:
                if use_packed_unchecked_launch_only:
                    op_name = (
                        "endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only"
                    )
                else:
                    op_name = (
                        "endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_launch_only_mse_vjp_direct_atomic_rgb_only"
                    )
            elif use_packed_rowdesc32_launch_only:
                if use_packed_unchecked_launch_only:
                    op_name = (
                        "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only"
                    )
                else:
                    op_name = (
                        "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_launch_only_mse_vjp_direct_atomic_rgb_only"
                    )
            elif use_packed_rowdesc_launch_only:
                if use_packed_unchecked_launch_only:
                    op_name = (
                        "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only"
                    )
                else:
                    op_name = (
                        "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_launch_only_mse_vjp_direct_atomic_rgb_only"
                    )
            elif use_packed_unchecked_launch_only:
                op_name = (
                    "endpoint_record_delta_replace_coeff16_packed_framegroup16_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only"
                )
            elif use_packed_launch_only:
                if use_packed_materialized:
                    op_name = (
                        "endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_launch_only_mse_vjp_direct_atomic_rgb_only"
                    )
                elif use_packed_recompute:
                    op_name = (
                        "endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_launch_only_mse_vjp_direct_atomic_rgb_only"
                    )
                elif use_packed_smallrun16:
                    op_name = (
                        "endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_launch_only_mse_vjp_direct_atomic_rgb_only"
                    )
                else:
                    op_name = (
                        "endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_mse_vjp_direct_atomic_rgb_only"
                    )
            elif use_packed_scalar:
                op_name = "endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only"
            elif use_packed_materialized:
                op_name = (
                    "endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only"
                )
            elif use_packed_recompute:
                op_name = (
                    "endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only"
                )
            elif use_packed_smallrun16:
                op_name = (
                    "endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only"
                )
            else:
                op_name = "endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only"
            ops = torch.ops.world_foam_lane2_fused_slab_v0
            if not hasattr(ops, op_name):
                raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
            if use_packed_rowdesc_launch_only or use_packed_rowdesc32_launch_only:
                if use_packed_unchecked_launch_only:
                    return getattr(ops, op_name)(
                        tape_device["delta_coeff_f16"],
                        tape_device["frame_t_f32"],
                        tape_device["row_begin_i32"],
                        tape_device["row_len_source_i16"],
                        tape_device["delta_base_record_i32"],
                        tape_device["delta_change_record_i32"],
                        site_rgba,
                        target_rgb_track,
                        tape_device["delta_config_i32"],
                        tape_device["delta_config_f32"],
                        int(tape_device["delta_launch_track_count"]),
                        int(tape_device["delta_launch_frame_count"]),
                        int(tape_device["delta_launch_site_count"]),
                    )
                return getattr(ops, op_name)(
                    tape_device["delta_coeff_f16"],
                    tape_device["frame_t_f32"],
                    tape_device["row_begin_i32"],
                    tape_device["row_len_source_i16"],
                    tape_device["delta_base_record_i32"],
                    tape_device["delta_change_record_i32"],
                    site_rgba,
                    target_rgb_track,
                    tape_device["delta_config_i32"],
                    tape_device["delta_config_f32"],
                    int(tape_device["delta_launch_boundary_count"]),
                    int(tape_device["delta_launch_track_count"]),
                    int(tape_device["delta_launch_frame_count"]),
                    int(tape_device["delta_launch_site_count"]),
                    int(tape_device["delta_launch_base_record_count"]),
                    int(tape_device["delta_launch_change_record_count"]),
                )
            return getattr(ops, op_name)(
                tape_device["delta_coeff_f16"],
                tape_device["frame_t_f32"],
                tape_device["base_offsets_i32"],
                tape_device["delta_base_record_i32"],
                tape_device["track_change_offsets_i32"],
                *((tape_device["track_chunk_change_offsets_i16"],) if not use_packed_scalar else ()),
                tape_device["change_frame_i32"],
                tape_device["change_offsets_i32"],
                tape_device["delta_change_record_i32"],
                site_rgba,
                target_rgb_track,
                tape_device["delta_config_i32"],
                tape_device["delta_config_f32"],
                *(
                    (
                        int(tape_device["delta_launch_track_count"]),
                        int(tape_device["delta_launch_frame_count"]),
                        int(tape_device["delta_launch_site_count"]),
                    )
                    if use_packed_unchecked_launch_only
                    else (
                        (
                            int(tape_device["delta_launch_boundary_count"]),
                            int(tape_device["delta_launch_track_count"]),
                            int(tape_device["delta_launch_frame_count"]),
                            int(tape_device["delta_launch_site_count"]),
                            int(tape_device["delta_launch_base_record_count"]),
                            int(tape_device["delta_launch_change_count"]),
                            int(tape_device["delta_launch_change_record_count"]),
                        )
                        if use_packed_launch_only
                        else ()
                    )
                ),
            )
        if use_packed_rowdesc_launch_only:
            raise ValueError("rowdesc launch-only packed delta requires delta_config_i32/delta_config_f32")
        if "change_offsets_i32" not in tape_device:
            raise ValueError("packed delta fused MSE path requires change_offsets_i32 unless rowdesc launch-only is active")
        if use_packed_unchecked_launch_only:
            raise ValueError("unchecked launch-only packed delta requires delta_config_i32/delta_config_f32")
        if use_packed_launch_only:
            raise ValueError("launch-only packed delta requires delta_config_i32/delta_config_f32")
        if use_packed_scalar:
            return endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only(
                tape_device["delta_coeff_f16"],
                tape_device["frame_t_f32"],
                tape_device["base_offsets_i32"],
                tape_device["delta_base_record_i32"],
                tape_device["track_change_offsets_i32"],
                tape_device["change_frame_i32"],
                tape_device["change_offsets_i32"],
                tape_device["delta_change_record_i32"],
                site_rgba,
                target_rgb_track,
                op_config,
                track_count=track_count,
                frame_count=frame_count,
                boundary_count=int(tape_device["delta_coeff_boundary_count"]),
            )
        if use_packed_materialized:
            packed_op = endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only
        elif use_packed_recompute:
            packed_op = endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only
        elif use_packed_smallrun16:
            packed_op = endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only
        else:
            packed_op = endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only
        return packed_op(
            tape_device["delta_coeff_f16"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i32"],
            tape_device["delta_base_record_i32"],
            tape_device["track_change_offsets_i32"],
            tape_device["track_chunk_change_offsets_i16"],
            tape_device["change_frame_i32"],
            tape_device["change_offsets_i32"],
            tape_device["delta_change_record_i32"],
            site_rgba,
            target_rgb_track,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["delta_coeff_boundary_count"]),
        )
    if "change_offsets_i32" not in tape_device:
        raise ValueError("delta-replace coeff16 fused MSE path requires change_offsets_i32")
    if "delta_base_record_i16x4" in tape_device and "delta_change_record_i16x4" in tape_device:
        is_framegroup = "delta_i16x4_framegroup16_fused_mse" in tape_device
        if "delta_config_i32" in tape_device and "delta_config_f32" in tape_device:
            _require_current_packed_endpoint_records_validated(
                tape_device=tape_device,
                site_count=int(site_rgba.shape[0]),
                track_count=int(track_count),
                frame_count=int(frame_count),
                runtime_device=site_rgba.device,
            )
            op_name = (
                "endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only"
                if is_framegroup
                else "endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only"
            )
            ops = torch.ops.world_foam_lane2_fused_slab_v0
            if not hasattr(ops, op_name):
                raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
            return getattr(ops, op_name)(
                tape_device["delta_coeff_f16"],
                tape_device["frame_t_f32"],
                tape_device["base_offsets_i32"],
                tape_device["delta_base_record_i16x4"],
                tape_device["track_change_offsets_i32"],
                *((tape_device["track_chunk_change_offsets_i16"],) if is_framegroup else ()),
                tape_device["change_frame_i32"],
                tape_device["change_offsets_i32"],
                tape_device["delta_change_record_i16x4"],
                site_rgba,
                target_rgb_track,
                tape_device["delta_config_i32"],
                tape_device["delta_config_f32"],
            )
        op = (
            endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only
            if is_framegroup
            else endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only
        )
        return op(
            tape_device["delta_coeff_f16"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i32"],
            tape_device["delta_base_record_i16x4"],
            tape_device["track_change_offsets_i32"],
            *((tape_device["track_chunk_change_offsets_i16"],) if is_framegroup else ()),
            tape_device["change_frame_i32"],
            tape_device["change_offsets_i32"],
            tape_device["delta_change_record_i16x4"],
            site_rgba,
            target_rgb_track,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["delta_coeff_boundary_count"]),
        )
    if "delta_base_record_i16cols" in tape_device and "delta_change_record_i16cols" in tape_device:
        if "delta_config_i32" not in tape_device or "delta_config_f32" not in tape_device:
            raise ValueError("i16cols packed delta fused MSE path requires delta_config_i32/delta_config_f32")
        _require_current_packed_endpoint_records_validated(
            tape_device=tape_device,
            site_count=int(site_rgba.shape[0]),
            track_count=int(track_count),
            frame_count=int(frame_count),
            runtime_device=site_rgba.device,
        )
        op_name = "endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only"
        ops = torch.ops.world_foam_lane2_fused_slab_v0
        if not hasattr(ops, op_name):
            raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
        return getattr(ops, op_name)(
            tape_device["delta_coeff_f16"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i32"],
            tape_device["delta_base_record_i16cols"],
            tape_device["track_change_offsets_i32"],
            tape_device["track_chunk_change_offsets_i16"],
            tape_device["change_frame_i32"],
            tape_device["change_offsets_i32"],
            tape_device["delta_change_record_i16cols"],
            site_rgba,
            target_rgb_track,
            tape_device["delta_config_i32"],
            tape_device["delta_config_f32"],
        )
    if "delta_base_record_i16x3" in tape_device and "delta_change_record_i16x3" in tape_device:
        use_i16x3_framegroup64 = "delta_i16x3_framegroup64_fused_mse" in tape_device
        use_i16x3_ownerreduce = "delta_i16x3_framegroup16_ownerreduce_fused_mse" in tape_device
        use_i16x3_materialized = "delta_i16x3_framegroup16_materialized_fused_mse" in tape_device
        use_i16x3_framegroup16 = (
            "delta_i16x3_framegroup16_fused_mse" in tape_device
            and not use_i16x3_framegroup64
            and not use_i16x3_ownerreduce
            and not use_i16x3_materialized
        )
        framegroup_extra_args = ()
        if use_i16x3_ownerreduce:
            framegroup_extra_args = (
                tape_device["track_chunk_change_offsets_i16"],
                tape_device["track_chunk_owner_offsets_i32"],
                tape_device["track_chunk_owner_i16"],
            )
        elif use_i16x3_framegroup64 or use_i16x3_framegroup16 or use_i16x3_materialized:
            framegroup_extra_args = (tape_device["track_chunk_change_offsets_i16"],)
        if "delta_config_i32" in tape_device and "delta_config_f32" in tape_device:
            _require_current_packed_endpoint_records_validated(
                tape_device=tape_device,
                site_count=int(site_rgba.shape[0]),
                track_count=int(track_count),
                frame_count=int(frame_count),
                runtime_device=site_rgba.device,
            )
            if use_i16x3_framegroup64:
                op_name = "endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only"
            elif use_i16x3_materialized:
                op_name = (
                    "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only"
                )
            elif use_i16x3_ownerreduce:
                op_name = (
                    "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only"
                )
            elif use_i16x3_framegroup16:
                op_name = "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only"
            else:
                op_name = "endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only"
            ops = torch.ops.world_foam_lane2_fused_slab_v0
            if not hasattr(ops, op_name):
                raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
            return getattr(ops, op_name)(
                tape_device["delta_coeff_f16"],
                tape_device["frame_t_f32"],
                tape_device["base_offsets_i32"],
                tape_device["delta_base_record_i16x3"],
                tape_device["track_change_offsets_i32"],
                *framegroup_extra_args,
                tape_device["change_frame_i32"],
                tape_device["change_offsets_i32"],
                tape_device["delta_change_record_i16x3"],
                site_rgba,
                target_rgb_track,
                tape_device["delta_config_i32"],
                tape_device["delta_config_f32"],
            )
        if use_i16x3_framegroup64:
            raise RuntimeError("i16x3 framegroup64 fused MSE path requires raw delta_config tensors")
        if use_i16x3_ownerreduce:
            op = endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only
        elif use_i16x3_materialized:
            op = endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only
        elif use_i16x3_framegroup16:
            op = endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only
        else:
            op = endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only
        return op(
            tape_device["delta_coeff_f16"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i32"],
            tape_device["delta_base_record_i16x3"],
            tape_device["track_change_offsets_i32"],
            *framegroup_extra_args,
            tape_device["change_frame_i32"],
            tape_device["change_offsets_i32"],
            tape_device["delta_change_record_i16x3"],
            site_rgba,
            target_rgb_track,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["delta_coeff_boundary_count"]),
        )
    if "delta_config_i32" in tape_device and "delta_config_f32" in tape_device:
        _require_current_packed_endpoint_records_validated(
            tape_device=tape_device,
            site_count=int(site_rgba.shape[0]),
            track_count=int(track_count),
            frame_count=int(frame_count),
            runtime_device=site_rgba.device,
        )
        op_name = "endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only"
        ops = torch.ops.world_foam_lane2_fused_slab_v0
        if not hasattr(ops, op_name):
            raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
        return getattr(ops, op_name)(
            tape_device["delta_coeff_f16"],
            tape_device["frame_t_f32"],
            tape_device["base_offsets_i32"],
            tape_device["base_owner_i32"],
            tape_device["base_left_i32"],
            tape_device["base_right_i32"],
            tape_device["track_change_offsets_i32"],
            tape_device["change_frame_i32"],
            tape_device["change_offsets_i32"],
            tape_device["change_owner_i32"],
            tape_device["change_left_i32"],
            tape_device["change_right_i32"],
            site_rgba,
            target_rgb_track,
            tape_device["delta_config_i32"],
            tape_device["delta_config_f32"],
        )
    return endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only(
        tape_device["delta_coeff_f16"],
        tape_device["frame_t_f32"],
        tape_device["base_offsets_i32"],
        tape_device["base_owner_i32"],
        tape_device["base_left_i32"],
        tape_device["base_right_i32"],
        tape_device["track_change_offsets_i32"],
        tape_device["change_frame_i32"],
        tape_device["change_offsets_i32"],
        tape_device["change_owner_i32"],
        tape_device["change_left_i32"],
        tape_device["change_right_i32"],
        site_rgba,
        target_rgb_track,
        op_config,
        track_count=track_count,
        frame_count=frame_count,
        boundary_count=int(tape_device["delta_coeff_boundary_count"]),
    )


def _delta_replace_coeff16_framegroup_fused_mse_objective(
    *,
    tape_device: dict[str, torch.Tensor],
    op_config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
) -> WorldFoamFrozenRGBMSEObjective:
    if (
        "delta_coeff_f16" not in tape_device
        or "delta_base_record_i16x3" not in tape_device
        or "delta_change_record_i16x3" not in tape_device
        or "track_chunk_change_offsets_i16" not in tape_device
    ):
        raise ValueError("framegroup fused MSE autograd path requires delta-replace i16x3 framegroup tape")
    world_foam_tape = dict(
        zip(
            FRAMEGROUP16_TAPE_KEYS,
            (
                tape_device["delta_coeff_f16"],
                tape_device["frame_t_f32"],
                tape_device["base_offsets_i32"],
                tape_device["delta_base_record_i16x3"],
                tape_device["track_change_offsets_i32"],
                tape_device["track_chunk_change_offsets_i16"],
                tape_device["change_frame_i32"],
                tape_device["change_offsets_i32"],
                tape_device["delta_change_record_i16x3"],
            ),
            strict=True,
        )
    )
    objective = WorldFoamFrozenRGBMSEObjective(
        tape=world_foam_tape,
        config=op_config,
        boundary_count=int(tape_device["delta_coeff_boundary_count"]),
        layout=WorldFoamTargetLayout.from_track_major(track_count=track_count, frame_count=frame_count),
        fused_loss_fn=promoted_framegroup16_loss_fn(),
    )
    return objective


def _block_coeff_fused_mse_loss_vjp(
    *,
    tape_device: dict[str, torch.Tensor],
    site_rgba: torch.Tensor,
    target_rgb_track: torch.Tensor,
    op_config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "block_anchor_offsets_i32" not in tape_device:
        raise ValueError("fused MSE path requires endpoint-record-edit block-coeff tape")
    coeff_key = "block_coeff_f16" if "block_coeff_f16" in tape_device else "block_coeff_f32"
    if coeff_key not in tape_device:
        raise ValueError("fused MSE path requires endpoint-record-edit block-coeff tape")
    if "block_anchor_record_i16x3" in tape_device and "block_op_record_i16x3" in tape_device:
        if coeff_key != "block_coeff_f16":
            raise ValueError("int16x3-record block-coeff fused MSE path requires coeff16 tape")
        if "block_coeff_config_i32" in tape_device and "block_coeff_config_f32" in tape_device:
            op_name = "endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only"
            ops = torch.ops.world_foam_lane2_fused_slab_v0
            if not hasattr(ops, op_name):
                raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
            return getattr(ops, op_name)(
                tape_device["block_coeff_f16"],
                tape_device["frame_t_f32"],
                tape_device["block_anchor_offsets_i32"],
                tape_device["block_anchor_record_i16x3"],
                tape_device["block_track_change_offsets_i32"],
                tape_device["block_change_frame_i32"],
                tape_device["block_op_offsets_i32"],
                tape_device["block_op_type_i32"],
                tape_device["block_op_pos_i32"],
                tape_device["block_op_record_i16x3"],
                site_rgba,
                target_rgb_track,
                tape_device["block_coeff_config_i32"],
                tape_device["block_coeff_config_f32"],
            )
        return endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
            tape_device["block_coeff_f16"],
            tape_device["frame_t_f32"],
            tape_device["block_anchor_offsets_i32"],
            tape_device["block_anchor_record_i16x3"],
            tape_device["block_track_change_offsets_i32"],
            tape_device["block_change_frame_i32"],
            tape_device["block_op_offsets_i32"],
            tape_device["block_op_type_i32"],
            tape_device["block_op_pos_i32"],
            tape_device["block_op_record_i16x3"],
            site_rgba,
            target_rgb_track,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["block_coeff_boundary_count"]),
            block_size=int(tape_device["block_size"]),
        )
    if "block_anchor_owner_i16" in tape_device and "block_op_owner_i16" in tape_device:
        if coeff_key != "block_coeff_f16":
            raise ValueError("int16-record block-coeff fused MSE path requires coeff16 tape")
        if "block_coeff_config_i32" in tape_device and "block_coeff_config_f32" in tape_device:
            op_name = "endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only"
            ops = torch.ops.world_foam_lane2_fused_slab_v0
            if not hasattr(ops, op_name):
                raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
            return getattr(ops, op_name)(
                tape_device["block_coeff_f16"],
                tape_device["frame_t_f32"],
                tape_device["block_anchor_offsets_i32"],
                tape_device["block_anchor_owner_i16"],
                tape_device["block_anchor_left_i16"],
                tape_device["block_anchor_right_i16"],
                tape_device["block_track_change_offsets_i32"],
                tape_device["block_change_frame_i32"],
                tape_device["block_op_offsets_i32"],
                tape_device["block_op_type_i32"],
                tape_device["block_op_pos_i32"],
                tape_device["block_op_owner_i16"],
                tape_device["block_op_left_i16"],
                tape_device["block_op_right_i16"],
                site_rgba,
                target_rgb_track,
                tape_device["block_coeff_config_i32"],
                tape_device["block_coeff_config_f32"],
            )
        return endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only(
            tape_device["block_coeff_f16"],
            tape_device["frame_t_f32"],
            tape_device["block_anchor_offsets_i32"],
            tape_device["block_anchor_owner_i16"],
            tape_device["block_anchor_left_i16"],
            tape_device["block_anchor_right_i16"],
            tape_device["block_track_change_offsets_i32"],
            tape_device["block_change_frame_i32"],
            tape_device["block_op_offsets_i32"],
            tape_device["block_op_type_i32"],
            tape_device["block_op_pos_i32"],
            tape_device["block_op_owner_i16"],
            tape_device["block_op_left_i16"],
            tape_device["block_op_right_i16"],
            site_rgba,
            target_rgb_track,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["block_coeff_boundary_count"]),
            block_size=int(tape_device["block_size"]),
        )
    if "block_anchor_record_i32" in tape_device and "block_op_record_i32" in tape_device:
        if coeff_key != "block_coeff_f16":
            raise ValueError("packed block-coeff fused MSE path requires coeff16 tape")
        if "block_coeff_config_i32" in tape_device and "block_coeff_config_f32" in tape_device:
            op_name = "endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only"
            ops = torch.ops.world_foam_lane2_fused_slab_v0
            if not hasattr(ops, op_name):
                raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
            return getattr(ops, op_name)(
                tape_device["block_coeff_f16"],
                tape_device["frame_t_f32"],
                tape_device["block_anchor_offsets_i32"],
                tape_device["block_anchor_record_i32"],
                tape_device["block_track_change_offsets_i32"],
                tape_device["block_change_frame_i32"],
                tape_device["block_op_offsets_i32"],
                tape_device["block_op_type_i32"],
                tape_device["block_op_pos_i32"],
                tape_device["block_op_record_i32"],
                site_rgba,
                target_rgb_track,
                tape_device["block_coeff_config_i32"],
                tape_device["block_coeff_config_f32"],
            )
        return endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only(
            tape_device["block_coeff_f16"],
            tape_device["frame_t_f32"],
            tape_device["block_anchor_offsets_i32"],
            tape_device["block_anchor_record_i32"],
            tape_device["block_track_change_offsets_i32"],
            tape_device["block_change_frame_i32"],
            tape_device["block_op_offsets_i32"],
            tape_device["block_op_type_i32"],
            tape_device["block_op_pos_i32"],
            tape_device["block_op_record_i32"],
            site_rgba,
            target_rgb_track,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["block_coeff_boundary_count"]),
            block_size=int(tape_device["block_size"]),
        )
    if "block_coeff_config_i32" in tape_device and "block_coeff_config_f32" in tape_device:
        op_name = (
            "endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only"
            if coeff_key == "block_coeff_f16"
            else "endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only"
        )
        ops = torch.ops.world_foam_lane2_fused_slab_v0
        if not hasattr(ops, op_name):
            raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
        return getattr(ops, op_name)(
            tape_device[coeff_key],
            tape_device["frame_t_f32"],
            tape_device["block_anchor_offsets_i32"],
            tape_device["block_anchor_owner_i32"],
            tape_device["block_anchor_left_i32"],
            tape_device["block_anchor_right_i32"],
            tape_device["block_track_change_offsets_i32"],
            tape_device["block_change_frame_i32"],
            tape_device["block_op_offsets_i32"],
            tape_device["block_op_type_i32"],
            tape_device["block_op_pos_i32"],
            tape_device["block_op_owner_i32"],
            tape_device["block_op_left_i32"],
            tape_device["block_op_right_i32"],
            site_rgba,
            target_rgb_track,
            tape_device["block_coeff_config_i32"],
            tape_device["block_coeff_config_f32"],
        )
    if coeff_key == "block_coeff_f16":
        return endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only(
            tape_device["block_coeff_f16"],
            tape_device["frame_t_f32"],
            tape_device["block_anchor_offsets_i32"],
            tape_device["block_anchor_owner_i32"],
            tape_device["block_anchor_left_i32"],
            tape_device["block_anchor_right_i32"],
            tape_device["block_track_change_offsets_i32"],
            tape_device["block_change_frame_i32"],
            tape_device["block_op_offsets_i32"],
            tape_device["block_op_type_i32"],
            tape_device["block_op_pos_i32"],
            tape_device["block_op_owner_i32"],
            tape_device["block_op_left_i32"],
            tape_device["block_op_right_i32"],
            site_rgba,
            target_rgb_track,
            op_config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(tape_device["block_coeff_boundary_count"]),
            block_size=int(tape_device["block_size"]),
        )
    return endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only(
        tape_device["block_coeff_f32"],
        tape_device["frame_t_f32"],
        tape_device["block_anchor_offsets_i32"],
        tape_device["block_anchor_owner_i32"],
        tape_device["block_anchor_left_i32"],
        tape_device["block_anchor_right_i32"],
        tape_device["block_track_change_offsets_i32"],
        tape_device["block_change_frame_i32"],
        tape_device["block_op_offsets_i32"],
        tape_device["block_op_type_i32"],
        tape_device["block_op_pos_i32"],
        tape_device["block_op_owner_i32"],
        tape_device["block_op_left_i32"],
        tape_device["block_op_right_i32"],
        site_rgba,
        target_rgb_track,
        op_config,
        track_count=track_count,
        frame_count=frame_count,
        boundary_count=int(tape_device["block_coeff_boundary_count"]),
        block_size=int(tape_device["block_size"]),
    )


def _adam_update(
    *,
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_index: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> None:
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bias_correction1 = 1.0 - beta1**step_index
    bias_correction2 = 1.0 - beta2**step_index
    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
    param.addcdiv_(exp_avg / bias_correction1, denom, value=-lr)
    param[:, :3].clamp_(0.0, 1.0)
    param[:, 3].clamp_(min=0.01)


def _prepare_owner_run_tapes(
    *,
    sites: tuple[Any, ...],
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    site_rgba: torch.Tensor,
    tape_mode: str,
    edit_block_size: int = 4,
    endpoint_record_source: str = "slow-owner-run",
    gate4_time_slabs: int = 1,
    gate4_residual_depth_padding: float = 0.001,
    experimental_native_cut_prep_delta: bool = False,
    experimental_native_sorted_delta: bool = False,
    experimental_minimal_packed_delta_device: bool = False,
    experimental_cpu_rebase_delta: bool = False,
    experimental_kernel_order_packed_delta_device: bool = False,
    experimental_smallrun16_packed_delta: bool = False,
    experimental_launch_only_packed_delta: bool = False,
    experimental_unchecked_launch_only_packed_delta: bool = False,
    experimental_reduce32_launch_only_packed_delta: bool = False,
    experimental_rowselect32_launch_only_packed_delta: bool = False,
    experimental_rowdesc_launch_only_packed_delta: bool = False,
    experimental_rowdesc32_launch_only_packed_delta: bool = False,
    experimental_native_pack_records: bool = False,
    experimental_native_emitted_pack_records: bool = False,
    experimental_selected_only_owner_run_delta_prep: bool = False,
    experimental_native_owner_run_cutwalk_delta: bool = False,
) -> dict[str, Any]:
    endpoint_record_modes = {
        "endpoint-record-edit",
        "endpoint-record-edit-fused-mse",
        "endpoint-record-edit-coeff16-fused-mse",
        "endpoint-record-delta-replace-coeff16-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x3-fused-mse",
        DELTA_I16X3_FRAMEGROUP16_MODE,
        DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse",
        DELTA_PACKED_SCALAR_MODE,
        DELTA_PACKED_FRAMEGROUP16_MODE,
        DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
        DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
        DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
        OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        DELTA_AUTO_FRAMEGROUP16_MODE,
        "endpoint-record-delta-replace-coeff16-i16x4-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
        "endpoint-record-edit-block4",
        "endpoint-record-edit-block-coeff",
        "endpoint-record-edit-block-coeff-rgb",
        "endpoint-record-edit-block-coeff-fused-mse",
        "endpoint-record-edit-block-coeff16",
        "endpoint-record-edit-block-coeff16-fused-mse",
        "endpoint-record-edit-block-coeff16-packed-fused-mse",
        "endpoint-record-edit-block-coeff16-i16-fused-mse",
        "endpoint-record-edit-block-coeff16-i16x3-fused-mse",
    }
    candidate_affine_mode = _is_gate4_affine_candidate_fused_mse_mode(tape_mode)
    if tape_mode not in {
        "owner-run",
        OWNER_RUN_FUSED_MSE_MODE,
        OWNER_RUN_FUSED_MSE_NOMID_MODE,
        "active-internal",
        "full",
        "endpoint-run",
        ENDPOINT_RUN_FUSED_MSE_MODE,
        *endpoint_record_modes,
        *GATE4_AFFINE_CANDIDATE_FUSED_MSE_MODES,
    }:
        raise ValueError(
            "tape_mode must be 'owner-run', "
            f"'{OWNER_RUN_FUSED_MSE_MODE}', '{OWNER_RUN_FUSED_MSE_NOMID_MODE}', "
            "'active-internal', 'full', 'endpoint-run', "
            f"'{ENDPOINT_RUN_FUSED_MSE_MODE}', "
            "'endpoint-record-edit', 'endpoint-record-edit-fused-mse', "
            "'endpoint-record-edit-coeff16-fused-mse', "
            "'endpoint-record-delta-replace-coeff16-fused-mse', "
            "'endpoint-record-delta-replace-coeff16-i16x3-fused-mse', "
            f"'{DELTA_I16X3_FRAMEGROUP16_MODE}', "
            f"'{DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE}', "
            "'endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse', "
            "'endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse', "
            "'endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse', "
            f"'{DELTA_PACKED_SCALAR_MODE}', "
            f"'{DELTA_PACKED_FRAMEGROUP16_MODE}', "
            f"'{DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE}', "
            f"'{DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE}', "
            f"'{DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE}', "
            f"'{OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE}', "
            f"'{OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE}', "
            f"'{OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE}', "
            f"'{DELTA_AUTO_FRAMEGROUP16_MODE}', "
            "'endpoint-record-delta-replace-coeff16-i16x4-fused-mse', "
            "'endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse', "
            "'endpoint-record-edit-block4', 'endpoint-record-edit-block-coeff', "
            "'endpoint-record-edit-block-coeff-rgb', 'endpoint-record-edit-block-coeff-fused-mse', "
            "'endpoint-record-edit-block-coeff16', 'endpoint-record-edit-block-coeff16-fused-mse', "
            "'endpoint-record-edit-block-coeff16-packed-fused-mse', "
            "'endpoint-record-edit-block-coeff16-i16-fused-mse', "
            "'endpoint-record-edit-block-coeff16-i16x3-fused-mse', "
            f"'{GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE}', "
            f"'{GATE4_AFFINE_CANDIDATE_NUM32_DEN16_TRACK_MSE_FUSED_MSE_MODE}', "
            f"'{GATE4_AFFINE_CANDIDATE_COEFF16_FUSED_MSE_MODE}', "
            f"'{GATE4_AFFINE_CANDIDATE_COEFF16_CAP224_FUSED_MSE_MODE}', "
            f"'{GATE4_AFFINE_CANDIDATE_COEFF16_DENSITYMASK_FUSED_MSE_MODE}', "
            f"'{GATE4_AFFINE_CANDIDATE_COEFF16_SAMPLE_REDUCE_FUSED_MSE_MODE}', "
            f"'{GATE4_AFFINE_CANDIDATE_COEFF16_SORTNET_FUSED_MSE_MODE}', "
            f"'{GATE4_AFFINE_CANDIDATE_COEFF16_FRAMEGROUP16_CACHED_FUSED_MSE_MODE}', "
            f"'{GATE4_AFFINE_CANDIDATE_COEFF16_SITECACHE_FUSED_MSE_MODE}', "
            f"'{GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_FUSED_MSE_MODE}', "
            f"'{GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_I16_FUSED_MSE_MODE}', "
            f"'{GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_FUSED_MSE_MODE}', "
            f"'{GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_I16_FUSED_MSE_MODE}', "
            f"or '{GATE4_AFFINE_CANDIDATE_COEFF16_TRACK_MSE_FUSED_MSE_MODE}'"
        )
    if endpoint_record_source not in {"slow-owner-run", "gate4-affine"}:
        raise ValueError("endpoint_record_source must be 'slow-owner-run' or 'gate4-affine'")
    if tape_mode in {
        OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    } and endpoint_record_source != "slow-owner-run":
        raise ValueError(f"{tape_mode} requires slow-owner-run source")
    if gate4_time_slabs <= 0:
        raise ValueError("gate4_time_slabs must be positive")
    boundaries = make_boundaries_4d(sites)
    prepare_timings: dict[str, float] = {}
    resolved_tape_mode = _resolve_delta_framegroup16_auto_mode(
        tape_mode,
        frame_count=frame_count,
        prefer_smallrun16=experimental_smallrun16_packed_delta,
    )
    delta_replace_modes = {
        "endpoint-record-delta-replace-coeff16-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x3-fused-mse",
        DELTA_I16X3_FRAMEGROUP16_MODE,
        DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse",
        DELTA_PACKED_SCALAR_MODE,
        DELTA_PACKED_FRAMEGROUP16_MODE,
        DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
        DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
        DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
        OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        "endpoint-record-delta-replace-coeff16-i16x4-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
    }
    owner_run_delta_packed_modes = {
        OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    }
    selected_only_owner_run_delta_prep = bool(
        experimental_selected_only_owner_run_delta_prep
        and endpoint_record_source == "slow-owner-run"
        and tape_mode in owner_run_delta_packed_modes
    )
    if experimental_selected_only_owner_run_delta_prep and not selected_only_owner_run_delta_prep:
        raise ValueError(
            "--experimental-selected-only-owner-run-delta-prep is only valid for slow-owner-run "
            "owner-run delta packed modes"
        )
    native_owner_run_cutwalk_delta = bool(
        experimental_native_owner_run_cutwalk_delta
        and endpoint_record_source == "slow-owner-run"
        and tape_mode in owner_run_delta_packed_modes
    )
    if experimental_native_owner_run_cutwalk_delta and not native_owner_run_cutwalk_delta:
        raise ValueError(
            "--experimental-native-owner-run-cutwalk-delta is only valid for slow-owner-run "
            "owner-run delta packed modes"
        )
    skip_baseline_segment_tapes = candidate_affine_mode or selected_only_owner_run_delta_prep or (
        endpoint_record_source == "gate4-affine"
        and tape_mode in endpoint_record_modes
        and resolved_tape_mode in delta_replace_modes
    )
    effective_native_emitted_pack_records = _effective_native_emitted_pack_records(
        requested=experimental_native_emitted_pack_records,
        resolved_tape_mode=resolved_tape_mode,
    )
    tape = None
    full = None
    active_internal = None
    owner_run = None
    endpoint_run = None
    if not skip_baseline_segment_tapes:
        phase_start = time.perf_counter()
        tape = build_segment_tape(
            sites=sites,
            boundaries=boundaries,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=frame_count,
            near=near,
            far=far,
            invalid_epsilon=invalid_epsilon,
            transmittance_threshold=transmittance_threshold,
        )
        prepare_timings["build_segment_tape_s"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        full = compact_segment_tape(tape)
        active_internal = compact_segment_tape(_with_counts(tape, tape.active_counts_i32))
        owner_run = compress_same_owner_runs(
            tape=tape,
            site_rgba_f32=site_rgba,
            transmittance_threshold=transmittance_threshold,
        )
        endpoint_run = compress_same_owner_endpoint_runs(tape)
        prepare_timings["compact_baseline_tapes_s"] = time.perf_counter() - phase_start
    endpoint_record_edit = None
    endpoint_record_delta_replace = None
    endpoint_record_block_edit = None
    endpoint_record_edit_device = None
    endpoint_record_delta_render_inputs: dict[str, torch.Tensor] | None = None
    coeff_f32 = None
    selected_extra_storage_bytes = 0
    selected_factorized_coeff_storage_bytes = 0
    gate4_tape = None
    gate4_endpoint_metadata: dict[str, Any] = {}
    use_kernel_order_packed_delta_device = False
    if candidate_affine_mode:
        phase_start = time.perf_counter()
        gate4_tape = build_gate4_affine_slab_tape(
            boundaries=boundaries,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=frame_count,
            time_slabs=gate4_time_slabs,
            near=near,
            far=far,
            invalid_epsilon=invalid_epsilon,
            residual_depth_padding=gate4_residual_depth_padding,
            layout="per-track",
            tile_h=1,
            tile_w=1,
            candidate_order="slab-mid-depth",
            sample_validation="skip",
        )
        prepare_timings["build_gate4_affine_candidate_tape_s"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        endpoint_record_edit_device = _move_gate4_affine_candidate_tape_to_mps(
            gate4_tape=gate4_tape,
            sites=sites,
            trackmse_fused_mse=_is_gate4_affine_candidate_trackmse_mode(tape_mode),
            coeff16_fused_mse=_is_gate4_affine_candidate_coeff16_mode(tape_mode),
            cap224_fused_mse=_is_gate4_affine_candidate_cap224_mode(tape_mode),
            densitymask_fused_mse=_is_gate4_affine_candidate_densitymask_mode(tape_mode),
            sample_reduce_fused_mse=_is_gate4_affine_candidate_sample_reduce_mode(tape_mode),
            sortnet_fused_mse=_is_gate4_affine_candidate_sortnet_mode(tape_mode),
            framegroup16_cached_fused_mse=_is_gate4_affine_candidate_framegroup16_cached_mode(tape_mode),
            sitecache_fused_mse=_is_gate4_affine_candidate_sitecache_mode(tape_mode),
            ownerupdate_fused_mse=_is_gate4_affine_candidate_ownerupdate_mode(tape_mode),
            ownerupdate_i16_fused_mse=_is_gate4_affine_candidate_ownerupdate_i16_mode(tape_mode),
            ownerkeep_fused_mse=_is_gate4_affine_candidate_ownerkeep_mode(tape_mode),
            ownerkeep_i16_fused_mse=_is_gate4_affine_candidate_ownerkeep_i16_mode(tape_mode),
        )
        prepare_timings["move_gate4_affine_candidate_tape_to_mps_s"] = time.perf_counter() - phase_start
        gate4_endpoint_metadata = {
            "time_slabs": int(gate4_time_slabs),
            "residual_depth_padding": float(gate4_residual_depth_padding),
            "candidate_count": int(gate4_tape.candidate_count),
            "candidate_replay_iterations": int(gate4_tape.candidate_replay_iterations),
            "missing_sample_events": int(gate4_tape.missing_sample_events),
            "extra_candidate_events": int(gate4_tape.extra_candidate_events),
            "sample_validation": str(gate4_tape.candidate_depth_order.get("sample_validation", "unknown")),
            "missing_sample_events_authoritative": bool(
                gate4_tape.candidate_depth_order.get("missing_sample_events_authoritative", False)
            ),
            "max_candidates_per_row": int(gate4_tape.max_candidates_per_row),
            "avg_candidates_per_row": float(gate4_tape.avg_candidates_per_row),
            "row_count": int(gate4_tape.row_count),
            "track_count": int(gate4_tape.track_count),
            "affine_candidate_csr_fused_mse": True,
            "affine_candidate_csr_trackmse_fused_mse": bool(_is_gate4_affine_candidate_trackmse_mode(tape_mode)),
            "affine_candidate_csr_coeff16_fused_mse": bool(_is_gate4_affine_candidate_coeff16_mode(tape_mode)),
            "affine_candidate_csr_cap224_fused_mse": bool(_is_gate4_affine_candidate_cap224_mode(tape_mode)),
            "affine_candidate_csr_densitymask_fused_mse": bool(
                _is_gate4_affine_candidate_densitymask_mode(tape_mode)
            ),
            "affine_candidate_csr_sample_reduce_fused_mse": bool(
                _is_gate4_affine_candidate_sample_reduce_mode(tape_mode)
            ),
            "affine_candidate_csr_sortnet_fused_mse": bool(_is_gate4_affine_candidate_sortnet_mode(tape_mode)),
            "affine_candidate_csr_framegroup16_cached_fused_mse": bool(
                _is_gate4_affine_candidate_framegroup16_cached_mode(tape_mode)
            ),
            "affine_candidate_csr_sitecache_fused_mse": bool(
                _is_gate4_affine_candidate_sitecache_mode(tape_mode)
            ),
            "affine_candidate_csr_ownerupdate_fused_mse": bool(
                _is_gate4_affine_candidate_ownerupdate_mode(tape_mode)
            ),
            "affine_candidate_csr_ownerupdate_i16_fused_mse": bool(
                _is_gate4_affine_candidate_ownerupdate_i16_mode(tape_mode)
            ),
            "affine_candidate_csr_ownerkeep_fused_mse": bool(_is_gate4_affine_candidate_ownerkeep_mode(tape_mode)),
            "affine_candidate_csr_ownerkeep_i16_fused_mse": bool(
                _is_gate4_affine_candidate_ownerkeep_i16_mode(tape_mode)
            ),
        }
    elif tape_mode in endpoint_record_modes:
        phase_start = time.perf_counter()
        sequences = None
        if endpoint_record_source == "gate4-affine":
            gate4_phase_start = time.perf_counter()
            gate4_tape = build_gate4_affine_slab_tape(
                boundaries=boundaries,
                rays=rays,
                frame_indices=frame_indices,
                frame_count=frame_count,
                time_slabs=gate4_time_slabs,
                near=near,
                far=far,
                invalid_epsilon=invalid_epsilon,
                residual_depth_padding=gate4_residual_depth_padding,
                layout="per-track",
                tile_h=1,
                tile_w=1,
                candidate_order="slab-mid-depth",
                sample_validation="skip",
            )
            prepare_timings["build_gate4_affine_endpoint_tape_s"] = time.perf_counter() - gate4_phase_start
            track_rays = gate4_tape.explicit_rays.reshape(gate4_tape.track_count, frame_count, 6).contiguous()
            frame_t = gate4_tape.frame_t.contiguous()
            if resolved_tape_mode in delta_replace_modes:
                gate4_delta_phase_start = time.perf_counter()
                endpoint_record_delta_replace = build_gate4_endpoint_delta_replace_tape(
                    tape=gate4_tape,
                    sites=sites,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                    experimental_native_cut_prep_delta=experimental_native_cut_prep_delta,
                    experimental_native_sorted_delta=experimental_native_sorted_delta,
                    experimental_native_emitted_pack_records=effective_native_emitted_pack_records,
                )
                prepare_timings["build_gate4_endpoint_delta_replace_tape_s"] = (
                    time.perf_counter() - gate4_delta_phase_start
                )
            else:
                gate4_sequence_phase_start = time.perf_counter()
                sequences = build_gate4_endpoint_run_sequences(
                    tape=gate4_tape,
                    sites=sites,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                )
                prepare_timings["build_gate4_endpoint_run_sequences_s"] = (
                    time.perf_counter() - gate4_sequence_phase_start
                )
            gate4_endpoint_metadata = {
                "time_slabs": int(gate4_time_slabs),
                "residual_depth_padding": float(gate4_residual_depth_padding),
                "candidate_count": int(gate4_tape.candidate_count),
                "candidate_replay_iterations": int(gate4_tape.candidate_replay_iterations),
                "missing_sample_events": int(gate4_tape.missing_sample_events),
                "extra_candidate_events": int(gate4_tape.extra_candidate_events),
                "sample_validation": str(gate4_tape.candidate_depth_order.get("sample_validation", "unknown")),
                "missing_sample_events_authoritative": bool(
                    gate4_tape.candidate_depth_order.get("missing_sample_events_authoritative", False)
                ),
                "max_candidates_per_row": int(gate4_tape.max_candidates_per_row),
                "avg_candidates_per_row": float(gate4_tape.avg_candidates_per_row),
                "experimental_native_cut_prep_delta": bool(experimental_native_cut_prep_delta),
                "experimental_native_sorted_delta": bool(experimental_native_sorted_delta),
                "experimental_native_pack_records": bool(experimental_native_pack_records),
                "experimental_native_emitted_pack_records": bool(experimental_native_emitted_pack_records),
                "experimental_native_emitted_pack_records_effective": bool(effective_native_emitted_pack_records),
            }
        else:
            if tape_mode in {
                OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            } and endpoint_record_source != "slow-owner-run":
                raise ValueError(f"{tape_mode} requires slow-owner-run source")
            if native_owner_run_cutwalk_delta:
                endpoint_record_delta_replace = _build_owner_run_delta_replace_native_cutwalk_tape(
                    sites=sites,
                    boundaries=boundaries,
                    rays=rays,
                    frame_indices=frame_indices,
                    frame_count=frame_count,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                    transmittance_threshold=transmittance_threshold,
                    site_rgba=site_rgba,
                )
            else:
                sequences, _sample_meta = _build_owner_run_sequences(
                    sites=sites,
                    boundaries=boundaries,
                    rays=rays,
                    frame_indices=frame_indices,
                    frame_count=frame_count,
                    near=near,
                    far=far,
                    invalid_epsilon=invalid_epsilon,
                    transmittance_threshold=(
                        transmittance_threshold
                        if tape_mode
                        in {
                            OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                            OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                            OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                            OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                        }
                        else 0.0
                    ),
                    site_rgba=site_rgba,
                    include_sample_meta=False,
                )
            track_rays, frame_t = _track_frame_rays(rays, frame_indices, frame_count=frame_count)
        prepare_timings["build_endpoint_record_sequences_s"] = time.perf_counter() - phase_start
        record_track_count = int(
            gate4_tape.track_count
            if gate4_tape is not None
            else tape.track_count
            if tape is not None
            else track_rays.shape[0]
        )
        if resolved_tape_mode not in delta_replace_modes:
            if sequences is None:
                raise RuntimeError("endpoint-record edit modes require materialized endpoint record sequences")
            phase_start = time.perf_counter()
            endpoint_record_edit = pack_endpoint_record_edit_tape(sequences, frame_count=frame_count)
            prepare_timings["pack_endpoint_record_edit_s"] = time.perf_counter() - phase_start
        else:
            prepare_timings["pack_endpoint_record_edit_s"] = 0.0
        if resolved_tape_mode in {
            "endpoint-record-delta-replace-coeff16-fused-mse",
            "endpoint-record-delta-replace-coeff16-i16x3-fused-mse",
            DELTA_I16X3_FRAMEGROUP16_MODE,
            DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
            "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
            "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse",
            "endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse",
            DELTA_PACKED_SCALAR_MODE,
            DELTA_PACKED_FRAMEGROUP16_MODE,
            DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
            DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
            DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
            OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            "endpoint-record-delta-replace-coeff16-i16x4-fused-mse",
            "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
        }:
            factorized_delta_mode = resolved_tape_mode in OWNER_RUN_DELTA_PACKED_FACTORIZED_MODES
            factorized_frame_select_mode = (
                resolved_tape_mode
                == OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE
            )
            factorized_frame_bitmask_mode = (
                resolved_tape_mode
                == OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE
            )
            if endpoint_record_delta_replace is None:
                if sequences is None:
                    raise RuntimeError("delta-replace modes require sequences or a direct Gate4 delta tape")
                phase_start = time.perf_counter()
                endpoint_record_delta_replace = pack_endpoint_record_delta_replace_tape(
                    sequences,
                    frame_count=frame_count,
                )
                prepare_timings["pack_endpoint_record_delta_replace_s"] = time.perf_counter() - phase_start
            else:
                prepare_timings["pack_endpoint_record_delta_replace_s"] = 0.0
            if experimental_cpu_rebase_delta:
                phase_start = time.perf_counter()
                endpoint_record_delta_replace = _rebase_endpoint_record_delta_replace_cpu(
                    endpoint_record_delta_replace
                )
                prepare_timings["cpu_rebase_endpoint_record_delta_replace_s"] = time.perf_counter() - phase_start
            else:
                prepare_timings["cpu_rebase_endpoint_record_delta_replace_s"] = 0.0
            phase_start = time.perf_counter()
            coeff_f32 = (
                None
                if factorized_delta_mode
                else (
                    build_gate4_boundary_depth_coefficients(tape=gate4_tape, boundaries=boundaries)
                    if gate4_tape is not None
                    else _track_boundary_coefficients(boundaries=boundaries, track_rays=track_rays, frame_t=frame_t)
                )
            )
            prepare_timings["build_endpoint_record_coefficients_s"] = time.perf_counter() - phase_start
            phase_start = time.perf_counter()
            use_minimal_packed_delta_device = bool(
                (
                    experimental_minimal_packed_delta_device
                    or resolved_tape_mode == OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE
                    or factorized_delta_mode
                )
                and _delta_mode_uses_packed_framegroup(resolved_tape_mode)
            )
            use_kernel_order_packed_delta_device = bool(
                experimental_kernel_order_packed_delta_device and _delta_mode_uses_packed_framegroup(resolved_tape_mode)
            )
            boundary_f32_cpu = _boundary_tensor(boundaries)
            if use_kernel_order_packed_delta_device:
                endpoint_record_edit_device = {}
                endpoint_record_delta_render_inputs = {
                    "boundary_f32": boundary_f32_cpu,
                    "rays_f32": track_rays,
                    "frame_t_f32": frame_t,
                }
            elif use_minimal_packed_delta_device:
                endpoint_record_edit_device = _move_endpoint_record_delta_replace_minimal_fused_tape_to_mps(
                    delta=endpoint_record_delta_replace,
                    frame_t_f32=frame_t,
                )
                endpoint_record_delta_render_inputs = {
                    "boundary_f32": boundary_f32_cpu,
                    "rays_f32": track_rays,
                    "frame_t_f32": frame_t,
                }
            else:
                endpoint_record_edit_device = _move_endpoint_record_delta_replace_tape_to_mps(
                    delta=endpoint_record_delta_replace,
                    boundary_f32=boundary_f32_cpu,
                    rays_f32=track_rays,
                    frame_t_f32=frame_t,
                )
            mps_device = torch.device("mps")
            if factorized_delta_mode:
                track_ray_coeff_f32_cpu = _track_ray_linear_coefficients(track_rays=track_rays, frame_t=frame_t)
                endpoint_record_edit_device["boundary_f32"] = boundary_f32_cpu.to(device=mps_device).contiguous()
                endpoint_record_edit_device["track_ray_coeff_f32"] = track_ray_coeff_f32_cpu.to(
                    device=mps_device
                ).contiguous()
                selected_factorized_coeff_storage_bytes = int(
                    boundary_f32_cpu.numel() * boundary_f32_cpu.element_size()
                    + track_ray_coeff_f32_cpu.numel() * track_ray_coeff_f32_cpu.element_size()
                )
            else:
                endpoint_record_edit_device["delta_coeff_f16"] = coeff_f32.to(
                    device=mps_device,
                    dtype=torch.float16,
                ).contiguous()
            endpoint_record_edit_device["delta_coeff_boundary_count"] = int(len(boundaries))
            endpoint_record_edit_device["delta_replace_fused_mse"] = True
            if resolved_tape_mode in {
                "endpoint-record-delta-replace-coeff16-i16x3-fused-mse",
                DELTA_I16X3_FRAMEGROUP16_MODE,
                DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
                "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
                "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse",
            }:
                endpoint_record_edit_device["delta_base_record_i16x3"] = _pack_endpoint_records_i16x3(
                    endpoint_record_delta_replace.base_owner_i32,
                    endpoint_record_delta_replace.base_left_i32,
                    endpoint_record_delta_replace.base_right_i32,
                    site_count=int(site_rgba.shape[0]),
                    boundary_count=int(len(boundaries)),
                ).to(device=mps_device).contiguous()
                endpoint_record_edit_device["delta_change_record_i16x3"] = _pack_endpoint_records_i16x3(
                    endpoint_record_delta_replace.change_owner_i32,
                    endpoint_record_delta_replace.change_left_i32,
                    endpoint_record_delta_replace.change_right_i32,
                    site_count=int(site_rgba.shape[0]),
                    boundary_count=int(len(boundaries)),
                ).to(device=mps_device).contiguous()
                if resolved_tape_mode in {
                    DELTA_I16X3_FRAMEGROUP16_MODE,
                    DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
                    "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
                    "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse",
                }:
                    if resolved_tape_mode == "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse":
                        endpoint_record_edit_device["delta_i16x3_framegroup64_fused_mse"] = torch.tensor(
                            [1],
                            device=mps_device,
                            dtype=torch.int32,
                        )
                    elif resolved_tape_mode == "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse":
                        endpoint_record_edit_device["delta_i16x3_framegroup16_ownerreduce_fused_mse"] = torch.tensor(
                            [1],
                            device=mps_device,
                            dtype=torch.int32,
                        )
                    elif resolved_tape_mode == DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE:
                        endpoint_record_edit_device["delta_i16x3_framegroup16_materialized_fused_mse"] = torch.tensor(
                            [1],
                            device=mps_device,
                            dtype=torch.int32,
                        )
                    else:
                        endpoint_record_edit_device["delta_i16x3_framegroup16_fused_mse"] = torch.tensor(
                            [1],
                            device=mps_device,
                            dtype=torch.int32,
                        )
                    chunk_offsets = build_delta_replace_chunk_change_offsets(
                        endpoint_record_delta_replace,
                        frame_count=frame_count,
                        chunk_size=64
                        if resolved_tape_mode == "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse"
                        else 16
                        if resolved_tape_mode == DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE
                        else 32,
                    )
                    if int(endpoint_record_edit_device["change_frame_i32"].shape[0]) > 32767:
                        raise ValueError("framegroup16 chunk offsets use int16 and require change count <= 32767")
                    endpoint_record_edit_device["track_chunk_change_offsets_i16"] = chunk_offsets.to(
                        device=mps_device,
                        dtype=torch.int16,
                    ).contiguous()
                    selected_extra_storage_bytes = int(chunk_offsets.numel() * 2)
                    if resolved_tape_mode == "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse":
                        owner_offsets, owner_ids = build_delta_replace_chunk_owner_lists(
                            endpoint_record_delta_replace,
                            frame_count=frame_count,
                            site_count=int(site_rgba.shape[0]),
                            chunk_size=32,
                        )
                        endpoint_record_edit_device["track_chunk_owner_offsets_i32"] = owner_offsets.to(
                            device=mps_device,
                            dtype=torch.int32,
                        ).contiguous()
                        endpoint_record_edit_device["track_chunk_owner_i16"] = owner_ids.to(
                            device=mps_device,
                            dtype=torch.int16,
                        ).contiguous()
                        selected_extra_storage_bytes += int(owner_offsets.numel() * 4 + owner_ids.numel() * 2)
            elif resolved_tape_mode == "endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse":
                endpoint_record_edit_device["delta_base_record_i16cols"] = _pack_endpoint_records_i16cols(
                    endpoint_record_delta_replace.base_owner_i32,
                    endpoint_record_delta_replace.base_left_i32,
                    endpoint_record_delta_replace.base_right_i32,
                    site_count=int(site_rgba.shape[0]),
                    boundary_count=int(len(boundaries)),
                ).to(device=mps_device).contiguous()
                endpoint_record_edit_device["delta_change_record_i16cols"] = _pack_endpoint_records_i16cols(
                    endpoint_record_delta_replace.change_owner_i32,
                    endpoint_record_delta_replace.change_left_i32,
                    endpoint_record_delta_replace.change_right_i32,
                    site_count=int(site_rgba.shape[0]),
                    boundary_count=int(len(boundaries)),
                ).to(device=mps_device).contiguous()
                endpoint_record_edit_device["delta_i16cols_framegroup16_fused_mse"] = torch.tensor(
                    [1],
                    device=mps_device,
                    dtype=torch.int32,
                )
                chunk_offsets = build_delta_replace_chunk_change_offsets(
                    endpoint_record_delta_replace,
                    frame_count=frame_count,
                )
                if int(endpoint_record_edit_device["change_frame_i32"].shape[0]) > 32767:
                    raise ValueError("framegroup16 chunk offsets use int16 and require change count <= 32767")
                endpoint_record_edit_device["track_chunk_change_offsets_i16"] = chunk_offsets.to(
                    device=mps_device,
                    dtype=torch.int16,
                ).contiguous()
                selected_extra_storage_bytes = int(chunk_offsets.numel() * 2)
            elif resolved_tape_mode in {
                DELTA_PACKED_SCALAR_MODE,
                DELTA_PACKED_FRAMEGROUP16_MODE,
                DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
                DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
                DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
                OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            }:
                if experimental_native_emitted_pack_records:
                    _validate_endpoint_record_components(
                        "native-emitted packed base endpoint record",
                        owner_i32=endpoint_record_delta_replace.base_owner_i32,
                        left_i32=endpoint_record_delta_replace.base_left_i32,
                        right_i32=endpoint_record_delta_replace.base_right_i32,
                        site_count=int(site_rgba.shape[0]),
                        boundary_count=int(len(boundaries)),
                    )
                    _validate_endpoint_record_components(
                        "native-emitted packed change endpoint record",
                        owner_i32=endpoint_record_delta_replace.change_owner_i32,
                        left_i32=endpoint_record_delta_replace.change_left_i32,
                        right_i32=endpoint_record_delta_replace.change_right_i32,
                        site_count=int(site_rgba.shape[0]),
                        boundary_count=int(len(boundaries)),
                    )
                    if endpoint_record_delta_replace.base_record_i32 is None:
                        raise ValueError("native-emitted packed records were requested but base_record_i32 is missing")
                    if endpoint_record_delta_replace.change_record_i32 is None:
                        raise ValueError("native-emitted packed records were requested but change_record_i32 is missing")
                    packed_base_record_i32 = endpoint_record_delta_replace.base_record_i32
                    packed_change_record_i32 = endpoint_record_delta_replace.change_record_i32
                else:
                    packed_base_record_i32 = _pack_endpoint_records_i32(
                        endpoint_record_delta_replace.base_owner_i32,
                        endpoint_record_delta_replace.base_left_i32,
                        endpoint_record_delta_replace.base_right_i32,
                        use_native=experimental_native_pack_records,
                        site_count=int(site_rgba.shape[0]),
                        boundary_count=int(len(boundaries)),
                    )
                    packed_change_record_i32 = _pack_endpoint_records_i32(
                        endpoint_record_delta_replace.change_owner_i32,
                        endpoint_record_delta_replace.change_left_i32,
                        endpoint_record_delta_replace.change_right_i32,
                        use_native=experimental_native_pack_records,
                        site_count=int(site_rgba.shape[0]),
                        boundary_count=int(len(boundaries)),
                    )
                packed_base_record_i32 = _validate_packed_endpoint_record_tensor(
                    "packed_base_record_i32",
                    packed_base_record_i32,
                    expected_shape=endpoint_record_delta_replace.base_owner_i32.shape,
                    site_count=int(site_rgba.shape[0]),
                    boundary_count=int(len(boundaries)),
                )
                packed_change_record_i32 = _validate_packed_endpoint_record_tensor(
                    "packed_change_record_i32",
                    packed_change_record_i32,
                    expected_shape=endpoint_record_delta_replace.change_owner_i32.shape,
                    site_count=int(site_rgba.shape[0]),
                    boundary_count=int(len(boundaries)),
                )
                delta_tables = _validate_endpoint_delta_index_tables(endpoint_record_delta_replace)
                change_count = int(delta_tables["change_frame_i32"].shape[0])
                frame_select_i16_cpu = (
                    _build_delta_frame_select_i16(endpoint_record_delta_replace, frame_count=frame_count)
                    if factorized_frame_select_mode
                    else None
                )
                frame_bitmask_i32_cpu = (
                    _build_delta_frame_bitmask_i32(endpoint_record_delta_replace, frame_count=frame_count)
                    if factorized_frame_bitmask_mode
                    else None
                )
                use_rowdesc_launch_only = bool(
                    experimental_launch_only_packed_delta
                    and experimental_rowdesc_launch_only_packed_delta
                    and resolved_tape_mode == DELTA_PACKED_FRAMEGROUP16_MODE
                )
                if use_kernel_order_packed_delta_device:
                    endpoint_record_edit_device["frame_t_f32"] = frame_t.to(device=mps_device).contiguous()
                    if not use_rowdesc_launch_only:
                        if factorized_delta_mode:
                            if factorized_frame_bitmask_mode:
                                endpoint_record_edit_device["base_offsets_i32"] = (
                                    delta_tables["base_offsets_i32"].to(device=mps_device).contiguous()
                                )
                            else:
                                endpoint_record_edit_device["base_offsets_i16"] = _to_mps_i16_meta_tensor(
                                    "base_offsets_i32",
                                    delta_tables["base_offsets_i32"],
                                    device=mps_device,
                                )
                        else:
                            endpoint_record_edit_device["base_offsets_i32"] = (
                                delta_tables["base_offsets_i32"].to(device=mps_device).contiguous()
                            )
                    endpoint_record_edit_device["delta_base_record_i32"] = packed_base_record_i32.to(
                        device=mps_device
                    ).contiguous()
                    if not use_rowdesc_launch_only:
                        if factorized_delta_mode:
                            if factorized_frame_select_mode:
                                if frame_select_i16_cpu is None:
                                    raise RuntimeError("frame-select factorized mode did not build frame_select_i16")
                                endpoint_record_edit_device["frame_change_index_i16"] = frame_select_i16_cpu.to(
                                    device=mps_device
                                ).contiguous()
                            elif factorized_frame_bitmask_mode:
                                if frame_bitmask_i32_cpu is None:
                                    raise RuntimeError("frame-bitmask factorized mode did not build frame_bitmask_i32")
                                endpoint_record_edit_device["track_change_offsets_i32"] = (
                                    delta_tables["track_change_offsets_i32"].to(device=mps_device).contiguous()
                                )
                                endpoint_record_edit_device["track_frame_mask_i32"] = frame_bitmask_i32_cpu.to(
                                    device=mps_device
                                ).contiguous()
                            else:
                                endpoint_record_edit_device["track_change_offsets_i16"] = _to_mps_i16_meta_tensor(
                                    "track_change_offsets_i32",
                                    delta_tables["track_change_offsets_i32"],
                                    device=mps_device,
                                )
                                endpoint_record_edit_device["change_frame_i16"] = _to_mps_i16_meta_tensor(
                                    "change_frame_i32",
                                    delta_tables["change_frame_i32"],
                                    device=mps_device,
                                )
                            if factorized_frame_bitmask_mode:
                                endpoint_record_edit_device["change_offsets_i32"] = (
                                    delta_tables["change_offsets_i32"].to(device=mps_device).contiguous()
                                )
                            else:
                                endpoint_record_edit_device["change_offsets_i16"] = _to_mps_i16_meta_tensor(
                                    "change_offsets_i32",
                                    delta_tables["change_offsets_i32"],
                                    device=mps_device,
                                )
                        else:
                            endpoint_record_edit_device["track_change_offsets_i32"] = (
                                delta_tables["track_change_offsets_i32"].to(device=mps_device).contiguous()
                            )
                            endpoint_record_edit_device["change_frame_i32"] = (
                                delta_tables["change_frame_i32"].to(device=mps_device).contiguous()
                            )
                            endpoint_record_edit_device["change_offsets_i32"] = (
                                delta_tables["change_offsets_i32"].to(device=mps_device).contiguous()
                            )
                    endpoint_record_edit_device["delta_change_record_i32"] = packed_change_record_i32.to(
                        device=mps_device
                    ).contiguous()
                else:
                    endpoint_record_edit_device["delta_base_record_i32"] = packed_base_record_i32.to(
                        device=mps_device
                    ).contiguous()
                    endpoint_record_edit_device["delta_change_record_i32"] = packed_change_record_i32.to(
                        device=mps_device
                    ).contiguous()
                    if factorized_delta_mode:
                        if factorized_frame_bitmask_mode:
                            endpoint_record_edit_device["base_offsets_i32"] = (
                                delta_tables["base_offsets_i32"].to(device=mps_device).contiguous()
                            )
                        else:
                            endpoint_record_edit_device["base_offsets_i16"] = _to_mps_i16_meta_tensor(
                                "base_offsets_i32",
                                delta_tables["base_offsets_i32"],
                                device=mps_device,
                            )
                        if factorized_frame_select_mode:
                            if frame_select_i16_cpu is None:
                                raise RuntimeError("frame-select factorized mode did not build frame_select_i16")
                            endpoint_record_edit_device["frame_change_index_i16"] = frame_select_i16_cpu.to(
                                device=mps_device
                            ).contiguous()
                        elif factorized_frame_bitmask_mode:
                            if frame_bitmask_i32_cpu is None:
                                raise RuntimeError("frame-bitmask factorized mode did not build frame_bitmask_i32")
                            endpoint_record_edit_device["track_change_offsets_i32"] = (
                                delta_tables["track_change_offsets_i32"].to(device=mps_device).contiguous()
                            )
                            endpoint_record_edit_device["track_frame_mask_i32"] = frame_bitmask_i32_cpu.to(
                                device=mps_device
                            ).contiguous()
                        else:
                            endpoint_record_edit_device["track_change_offsets_i16"] = _to_mps_i16_meta_tensor(
                                "track_change_offsets_i32",
                                delta_tables["track_change_offsets_i32"],
                                device=mps_device,
                            )
                            endpoint_record_edit_device["change_frame_i16"] = _to_mps_i16_meta_tensor(
                                "change_frame_i32",
                                delta_tables["change_frame_i32"],
                                device=mps_device,
                            )
                        if factorized_frame_bitmask_mode:
                            endpoint_record_edit_device["change_offsets_i32"] = (
                                delta_tables["change_offsets_i32"].to(device=mps_device).contiguous()
                            )
                        else:
                            endpoint_record_edit_device["change_offsets_i16"] = _to_mps_i16_meta_tensor(
                                "change_offsets_i32",
                                delta_tables["change_offsets_i32"],
                                device=mps_device,
                            )
                if resolved_tape_mode == DELTA_PACKED_SCALAR_MODE:
                    flag_name = "delta_packed_scalar_fused_mse"
                elif resolved_tape_mode == DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE:
                    flag_name = "delta_packed_framegroup16_materialized_fused_mse"
                elif resolved_tape_mode in {
                    DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
                    OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                }:
                    flag_name = "delta_packed_framegroup16_recompute_fused_mse"
                elif resolved_tape_mode == OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE:
                    flag_name = "delta_packed_framegroup16_factorized_recompute_fused_mse"
                elif (
                    resolved_tape_mode
                    == OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE
                ):
                    flag_name = "delta_packed_frameselect_factorized_recompute_fused_mse"
                elif (
                    resolved_tape_mode
                    == OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE
                ):
                    flag_name = "delta_packed_framebitmask_factorized_recompute_fused_mse"
                elif resolved_tape_mode == DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE:
                    flag_name = "delta_packed_framegroup16_smallrun16_fused_mse"
                else:
                    flag_name = "delta_packed_framegroup16_fused_mse"
                if use_kernel_order_packed_delta_device:
                    endpoint_record_edit_device[flag_name] = True
                else:
                    endpoint_record_edit_device[flag_name] = torch.tensor(
                        [1],
                        device=mps_device,
                        dtype=torch.int32,
                    )
                if experimental_launch_only_packed_delta and _delta_mode_uses_packed_framegroup(resolved_tape_mode):
                    endpoint_record_edit_device["delta_packed_framegroup16_launch_only_fused_mse"] = True
                    if experimental_unchecked_launch_only_packed_delta:
                        if resolved_tape_mode != DELTA_PACKED_FRAMEGROUP16_MODE:
                            raise ValueError(
                                "unchecked launch-only packed delta is only implemented for the default "
                                "packed framegroup16 fused-MSE shader"
                            )
                        endpoint_record_edit_device[
                            "delta_packed_framegroup16_unchecked_launch_only_fused_mse"
                        ] = True
                    if experimental_reduce32_launch_only_packed_delta:
                        if resolved_tape_mode != DELTA_PACKED_FRAMEGROUP16_MODE:
                            raise ValueError(
                                "reduce32 launch-only packed delta is only implemented for the default "
                                "packed framegroup16 fused-MSE shader"
                            )
                        endpoint_record_edit_device[
                            "delta_packed_framegroup16_reduce32_launch_only_fused_mse"
                        ] = True
                    if experimental_rowselect32_launch_only_packed_delta:
                        if resolved_tape_mode != DELTA_PACKED_FRAMEGROUP16_MODE:
                            raise ValueError(
                                "rowselect32 launch-only packed delta is only implemented for the default "
                                "packed framegroup16 fused-MSE shader"
                            )
                        endpoint_record_edit_device[
                            "delta_packed_framegroup16_rowselect32_launch_only_fused_mse"
                        ] = True
                    if experimental_rowdesc_launch_only_packed_delta:
                        if resolved_tape_mode != DELTA_PACKED_FRAMEGROUP16_MODE:
                            raise ValueError(
                                "rowdesc launch-only packed delta is only implemented for the default "
                                "packed framegroup16 fused-MSE shader"
                            )
                        row_begin_i32, row_len_source_i16 = build_delta_replace_frame_row_descriptors(
                            endpoint_record_delta_replace,
                            frame_count=frame_count,
                        )
                        endpoint_record_edit_device[
                            "delta_packed_framegroup16_rowdesc_launch_only_fused_mse"
                        ] = True
                        if experimental_rowdesc32_launch_only_packed_delta:
                            endpoint_record_edit_device[
                                "delta_packed_framegroup16_rowdesc32_launch_only_fused_mse"
                            ] = True
                        endpoint_record_edit_device["row_begin_i32"] = row_begin_i32.to(
                            device=mps_device,
                            dtype=torch.int32,
                        ).contiguous()
                        endpoint_record_edit_device["row_len_source_i16"] = row_len_source_i16.to(
                            device=mps_device,
                            dtype=torch.int16,
                        ).contiguous()
                        selected_extra_storage_bytes = int(
                            row_begin_i32.numel() * 4 + row_len_source_i16.numel() * 2
                        )
                endpoint_record_edit_device["delta_launch_boundary_count"] = int(len(boundaries))
                endpoint_record_edit_device["delta_launch_track_count"] = int(record_track_count)
                endpoint_record_edit_device["delta_launch_frame_count"] = int(frame_count)
                endpoint_record_edit_device["delta_launch_site_count"] = int(site_rgba.shape[0])
                endpoint_record_edit_device["delta_launch_base_record_count"] = int(packed_base_record_i32.shape[0])
                endpoint_record_edit_device["delta_launch_change_count"] = change_count
                endpoint_record_edit_device["delta_launch_change_record_count"] = int(packed_change_record_i32.shape[0])
                if (
                    resolved_tape_mode != DELTA_PACKED_SCALAR_MODE
                    and not factorized_frame_select_mode
                    and not factorized_frame_bitmask_mode
                    and not bool(endpoint_record_edit_device.get("delta_packed_framegroup16_rowdesc_launch_only_fused_mse"))
                ):
                    chunk_offsets = build_delta_replace_chunk_change_offsets(
                        endpoint_record_delta_replace,
                        frame_count=frame_count,
                        chunk_size=16 if resolved_tape_mode == DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE else 32,
                    )
                    if change_count > 32767:
                        raise ValueError("framegroup16 chunk offsets use int16 and require change count <= 32767")
                    endpoint_record_edit_device["track_chunk_change_offsets_i16"] = chunk_offsets.to(
                        device=mps_device,
                        dtype=torch.int16,
                    ).contiguous()
                    selected_extra_storage_bytes = int(chunk_offsets.numel() * 2)
                    if factorized_delta_mode:
                        for dense_meta_key in (
                            "base_offsets_i32",
                            "track_change_offsets_i32",
                            "change_frame_i32",
                            "change_offsets_i32",
                        ):
                            endpoint_record_edit_device.pop(dense_meta_key, None)
                if factorized_frame_select_mode:
                    if frame_select_i16_cpu is None:
                        raise RuntimeError("frame-select factorized mode did not build frame_select_i16")
                    selected_extra_storage_bytes = int(frame_select_i16_cpu.numel() * 2)
                    for dense_meta_key in (
                        "base_offsets_i32",
                        "track_change_offsets_i32",
                        "change_frame_i32",
                        "change_offsets_i32",
                        "track_chunk_change_offsets_i16",
                    ):
                        endpoint_record_edit_device.pop(dense_meta_key, None)
                if factorized_frame_bitmask_mode:
                    if frame_bitmask_i32_cpu is None:
                        raise RuntimeError("frame-bitmask factorized mode did not build frame_bitmask_i32")
                    selected_extra_storage_bytes = int(frame_bitmask_i32_cpu.numel() * 4)
                    for dense_meta_key in (
                        "change_frame_i32",
                        "track_chunk_change_offsets_i16",
                    ):
                        endpoint_record_edit_device.pop(dense_meta_key, None)
            elif resolved_tape_mode in {
                "endpoint-record-delta-replace-coeff16-i16x4-fused-mse",
                "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
            }:
                endpoint_record_edit_device["delta_base_record_i16x4"] = _pack_endpoint_records_i16x4(
                    endpoint_record_delta_replace.base_owner_i32,
                    endpoint_record_delta_replace.base_left_i32,
                    endpoint_record_delta_replace.base_right_i32,
                    site_count=int(site_rgba.shape[0]),
                    boundary_count=int(len(boundaries)),
                ).to(device=mps_device).contiguous()
                endpoint_record_edit_device["delta_change_record_i16x4"] = _pack_endpoint_records_i16x4(
                    endpoint_record_delta_replace.change_owner_i32,
                    endpoint_record_delta_replace.change_left_i32,
                    endpoint_record_delta_replace.change_right_i32,
                    site_count=int(site_rgba.shape[0]),
                    boundary_count=int(len(boundaries)),
                ).to(device=mps_device).contiguous()
                if resolved_tape_mode == "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse":
                    endpoint_record_edit_device["delta_i16x4_framegroup16_fused_mse"] = torch.tensor(
                        [1],
                        device=mps_device,
                        dtype=torch.int32,
                    )
                    chunk_offsets = build_delta_replace_chunk_change_offsets(
                        endpoint_record_delta_replace,
                        frame_count=frame_count,
                    )
                    if int(endpoint_record_edit_device["change_frame_i32"].shape[0]) > 32767:
                        raise ValueError("framegroup16 chunk offsets use int16 and require change count <= 32767")
                    endpoint_record_edit_device["track_chunk_change_offsets_i16"] = chunk_offsets.to(
                        device=mps_device,
                        dtype=torch.int16,
                    ).contiguous()
                    selected_extra_storage_bytes = int(chunk_offsets.numel() * 2)
            endpoint_record_edit_device["delta_config_i32"] = torch.tensor(
                [
                    int(len(boundaries)),
                    record_track_count,
                    int(frame_count),
                    int(site_rgba.shape[0]),
                    int(endpoint_record_delta_replace.base_owner_i32.shape[0]),
                    int(endpoint_record_delta_replace.change_frame_i32.shape[0]),
                    int(endpoint_record_delta_replace.change_owner_i32.shape[0]),
                ]
                + (
                    [int(endpoint_record_edit_device["track_chunk_owner_i16"].shape[0])]
                    if "delta_i16x3_framegroup16_ownerreduce_fused_mse" in endpoint_record_edit_device
                    else []
                ),
                device=mps_device,
                dtype=torch.int32,
            )
            endpoint_record_edit_device["delta_config_f32"] = torch.tensor(
                [near, far, invalid_epsilon, transmittance_threshold],
                device=mps_device,
                dtype=torch.float32,
            )
            endpoint_record_edit_device["delta_packed_records_validated"] = (
                _packed_endpoint_direct_config_validation_marker(
                    tape_device=endpoint_record_edit_device,
                    site_count=int(site_rgba.shape[0]),
                    track_count=int(record_track_count),
                    frame_count=int(frame_count),
                )
            )
            prepare_timings["move_endpoint_record_delta_replace_to_mps_s"] = time.perf_counter() - phase_start
        elif tape_mode in {
            "endpoint-record-edit-block4",
            "endpoint-record-edit-block-coeff",
            "endpoint-record-edit-block-coeff-rgb",
            "endpoint-record-edit-block-coeff-fused-mse",
            "endpoint-record-edit-block-coeff16",
            "endpoint-record-edit-block-coeff16-fused-mse",
            "endpoint-record-edit-block-coeff16-packed-fused-mse",
            "endpoint-record-edit-block-coeff16-i16-fused-mse",
            "endpoint-record-edit-block-coeff16-i16x3-fused-mse",
        }:
            endpoint_record_block_edit = pack_endpoint_record_block_edit_tape(
                sequences,
                frame_count=frame_count,
                block_size=edit_block_size,
            )
            coeff_f32 = (
                _track_boundary_coefficients(boundaries=boundaries, track_rays=track_rays, frame_t=frame_t)
                if tape_mode
                in {
                    "endpoint-record-edit-block-coeff",
                    "endpoint-record-edit-block-coeff-rgb",
                    "endpoint-record-edit-block-coeff-fused-mse",
                    "endpoint-record-edit-block-coeff16",
                    "endpoint-record-edit-block-coeff16-fused-mse",
                    "endpoint-record-edit-block-coeff16-packed-fused-mse",
                    "endpoint-record-edit-block-coeff16-i16-fused-mse",
                    "endpoint-record-edit-block-coeff16-i16x3-fused-mse",
                }
                else None
            )
            endpoint_record_edit_device = _move_endpoint_record_block4_tape_to_mps(
                edit=endpoint_record_edit,
                block_edit=endpoint_record_block_edit,
                boundary_f32=_boundary_tensor(boundaries),
                rays_f32=track_rays,
                frame_t_f32=frame_t,
                coeff_f32=coeff_f32
                if tape_mode
                in {
                    "endpoint-record-edit-block-coeff",
                    "endpoint-record-edit-block-coeff-rgb",
                    "endpoint-record-edit-block-coeff-fused-mse",
                }
                else None,
                coeff_f16=coeff_f32
                if tape_mode
                in {
                    "endpoint-record-edit-block-coeff16",
                    "endpoint-record-edit-block-coeff16-fused-mse",
                    "endpoint-record-edit-block-coeff16-packed-fused-mse",
                    "endpoint-record-edit-block-coeff16-i16-fused-mse",
                    "endpoint-record-edit-block-coeff16-i16x3-fused-mse",
                }
                else None,
                packed_records=tape_mode == "endpoint-record-edit-block-coeff16-packed-fused-mse",
                i16_records=tape_mode == "endpoint-record-edit-block-coeff16-i16-fused-mse",
                i16x3_records=tape_mode == "endpoint-record-edit-block-coeff16-i16x3-fused-mse",
            )
            endpoint_record_edit_device["block_coeff_rgb_only"] = tape_mode in {
                "endpoint-record-edit-block-coeff-rgb",
                "endpoint-record-edit-block-coeff-fused-mse",
            }
            endpoint_record_edit_device["block_coeff_fused_mse"] = (
                tape_mode
                in {
                    "endpoint-record-edit-block-coeff-fused-mse",
                    "endpoint-record-edit-block-coeff16-fused-mse",
                    "endpoint-record-edit-block-coeff16-packed-fused-mse",
                    "endpoint-record-edit-block-coeff16-i16-fused-mse",
                    "endpoint-record-edit-block-coeff16-i16x3-fused-mse",
                }
            )
            if "block_coeff_f32" in endpoint_record_edit_device or "block_coeff_f16" in endpoint_record_edit_device:
                block_count = (frame_count + edit_block_size - 1) // edit_block_size
                mps_device = torch.device("mps")
                endpoint_record_edit_device["block_coeff_config_i32"] = torch.tensor(
                    [
                        int(endpoint_record_edit_device["block_coeff_boundary_count"]),
                        record_track_count,
                        int(frame_count),
                        int(site_rgba.shape[0]),
                        int(endpoint_record_edit_device["block_anchor_owner_i32"].shape[0]),
                        int(endpoint_record_edit_device["block_change_frame_i32"].shape[0]),
                        int(endpoint_record_edit_device["block_op_type_i32"].shape[0]),
                        int(edit_block_size),
                        int(block_count),
                    ],
                    device=mps_device,
                    dtype=torch.int32,
                )
                endpoint_record_edit_device["block_coeff_config_f32"] = torch.tensor(
                    [near, far, invalid_epsilon, transmittance_threshold],
                    device=mps_device,
                    dtype=torch.float32,
                )
        else:
            endpoint_record_edit_device = _move_endpoint_record_edit_tape_to_mps(
                edit=endpoint_record_edit,
                boundary_f32=_boundary_tensor(boundaries),
                rays_f32=track_rays,
                frame_t_f32=frame_t,
            )
            if tape_mode == "endpoint-record-edit-coeff16-fused-mse":
                coeff_f32 = _track_boundary_coefficients(boundaries=boundaries, track_rays=track_rays, frame_t=frame_t)
                endpoint_record_edit_device["edit_coeff_f16"] = coeff_f32.to(
                    device=torch.device("mps"),
                    dtype=torch.float16,
                ).contiguous()
                endpoint_record_edit_device["edit_coeff_boundary_count"] = int(len(boundaries))
            endpoint_record_edit_device["edit_fused_mse"] = tape_mode in {
                "endpoint-record-edit-fused-mse",
                "endpoint-record-edit-coeff16-fused-mse",
            }
            if tape_mode in {"endpoint-record-edit-fused-mse", "endpoint-record-edit-coeff16-fused-mse"}:
                mps_device = torch.device("mps")
                endpoint_record_edit_device["edit_config_i32"] = torch.tensor(
                    [
                        int(endpoint_record_edit_device["boundary_f32"].shape[0]),
                        record_track_count,
                        int(frame_count),
                        int(site_rgba.shape[0]),
                        int(endpoint_record_edit_device["base_owner_i32"].shape[0]),
                        int(endpoint_record_edit_device["change_frame_i32"].shape[0]),
                        int(endpoint_record_edit_device["op_type_i32"].shape[0]),
                    ],
                    device=mps_device,
                    dtype=torch.int32,
                )
                endpoint_record_edit_device["edit_config_f32"] = torch.tensor(
                    [near, far, invalid_epsilon, transmittance_threshold],
                    device=mps_device,
                    dtype=torch.float32,
                )
    if candidate_affine_mode:
        if gate4_tape is None or endpoint_record_edit_device is None:
            raise RuntimeError("Gate4 affine candidate mode requires a prepared affine candidate tape")
        selected = None
        selected_device = endpoint_record_edit_device
        selected_segments = int(gate4_tape.candidate_count)
        selected_storage_bytes = sum(
            _tensor_storage_bytes(value)
            for value in selected_device.values()
            if isinstance(value, torch.Tensor)
        )
        baseline_segments = max(selected_segments, 1)
        baseline_storage_bytes = max(selected_storage_bytes, 1)
        active_internal_segments = 0
        owner_run_segments = 0
        endpoint_run_segments = selected_segments
        active_internal_storage_bytes = 0
        owner_run_storage_bytes = 0
        endpoint_run_storage_bytes = selected_storage_bytes
        max_selected_segments = int(gate4_tape.max_candidates_per_row)
        prepared_track_count = int(gate4_tape.track_count)
    elif skip_baseline_segment_tapes:
        if endpoint_record_delta_replace is None or endpoint_record_edit_device is None:
            raise RuntimeError("fast Gate4 endpoint-record path requires a delta-replace tape")
        base_lengths = (
            endpoint_record_delta_replace.base_offsets_i32[1:]
            - endpoint_record_delta_replace.base_offsets_i32[:-1]
        )
        change_lengths = (
            endpoint_record_delta_replace.change_offsets_i32[1:]
            - endpoint_record_delta_replace.change_offsets_i32[:-1]
        )
        row_lengths = torch.cat((base_lengths, change_lengths)) if change_lengths.numel() else base_lengths
        if selected_only_owner_run_delta_prep:
            _row_begin_i32, row_len_source_i16 = build_delta_replace_frame_row_descriptors(
                endpoint_record_delta_replace,
                frame_count=frame_count,
            )
            expanded_row_lengths = row_len_source_i16.to(dtype=torch.int32).bitwise_and(0x3FFF)
            max_selected_segments = int(expanded_row_lengths.max().item()) if expanded_row_lengths.numel() else 0
            selected_segments = int(expanded_row_lengths.sum().item())
        else:
            max_selected_segments = int(row_lengths.max().item()) if row_lengths.numel() else 0
            selected_segments = int(
                endpoint_record_delta_replace.base_owner_i32.numel()
                + endpoint_record_delta_replace.change_owner_i32.numel()
            )
        if (
            resolved_tape_mode == DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE
            and max_selected_segments > DELTA_SMALLRUN16_MAX_SEGMENTS
        ):
            raise ValueError(
                f"{DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE} requires max row segments <= "
                f"{DELTA_SMALLRUN16_MAX_SEGMENTS}, got {max_selected_segments}"
            )
        selected = None
        selected_device = endpoint_record_edit_device
        selected_uses_rowdesc_storage = bool(
            selected_device.get("delta_packed_framegroup16_rowdesc_launch_only_fused_mse", False)
        )
        selected_storage_bytes = _selected_tape_storage_bytes(
            tape_mode=tape_mode,
            selected=endpoint_record_delta_replace,
            endpoint_record_edit=endpoint_record_edit,
            endpoint_record_delta_replace=endpoint_record_delta_replace,
            endpoint_record_block_edit=endpoint_record_block_edit,
            coeff_f32=coeff_f32,
            extra_storage_bytes=selected_extra_storage_bytes,
            factorized_coeff_storage_bytes=selected_factorized_coeff_storage_bytes,
            record_bytes_override=(
                _delta_framegroup_record_bytes(resolved_tape_mode)
                if tape_mode == DELTA_AUTO_FRAMEGROUP16_MODE
                else None
            ),
            include_delta_index_storage=not selected_uses_rowdesc_storage,
        )
        baseline_segments = max(selected_segments, 1)
        baseline_storage_bytes = max(selected_storage_bytes, 1)
        active_internal_segments = 0
        owner_run_segments = selected_segments if selected_only_owner_run_delta_prep else 0
        endpoint_run_segments = selected_segments
        active_internal_storage_bytes = 0
        owner_run_storage_bytes = selected_storage_bytes if selected_only_owner_run_delta_prep else 0
        endpoint_run_storage_bytes = selected_storage_bytes
        prepared_track_count = int(record_track_count)
    else:
        selected = {
            "owner-run": owner_run,
            OWNER_RUN_FUSED_MSE_MODE: owner_run,
            OWNER_RUN_FUSED_MSE_NOMID_MODE: owner_run,
            "active-internal": active_internal,
            "full": full,
            "endpoint-run": endpoint_run,
            ENDPOINT_RUN_FUSED_MSE_MODE: endpoint_run,
            "endpoint-record-edit": endpoint_run,
            "endpoint-record-edit-fused-mse": endpoint_run,
            "endpoint-record-edit-coeff16-fused-mse": endpoint_run,
            "endpoint-record-delta-replace-coeff16-fused-mse": endpoint_run,
            "endpoint-record-delta-replace-coeff16-i16x3-fused-mse": endpoint_run,
            DELTA_I16X3_FRAMEGROUP16_MODE: endpoint_run,
            DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE: endpoint_run,
            "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse": endpoint_run,
            "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse": endpoint_run,
            "endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse": endpoint_run,
            DELTA_PACKED_SCALAR_MODE: endpoint_run,
            DELTA_PACKED_FRAMEGROUP16_MODE: endpoint_run,
            DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE: endpoint_run,
            DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE: endpoint_run,
            DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE: endpoint_run,
            OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE: owner_run,
            OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE: owner_run,
            OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE: owner_run,
            OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE: owner_run,
            DELTA_AUTO_FRAMEGROUP16_MODE: endpoint_run,
            "endpoint-record-delta-replace-coeff16-i16x4-fused-mse": endpoint_run,
            "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse": endpoint_run,
            "endpoint-record-edit-block4": endpoint_run,
            "endpoint-record-edit-block-coeff": endpoint_run,
            "endpoint-record-edit-block-coeff-rgb": endpoint_run,
            "endpoint-record-edit-block-coeff-fused-mse": endpoint_run,
            "endpoint-record-edit-block-coeff16": endpoint_run,
            "endpoint-record-edit-block-coeff16-fused-mse": endpoint_run,
            "endpoint-record-edit-block-coeff16-packed-fused-mse": endpoint_run,
            "endpoint-record-edit-block-coeff16-i16-fused-mse": endpoint_run,
            "endpoint-record-edit-block-coeff16-i16x3-fused-mse": endpoint_run,
        }[tape_mode]
        if selected is None or full is None or active_internal is None or owner_run is None or endpoint_run is None:
            raise RuntimeError("baseline segment tapes were not built")
        row_lengths = selected.offsets_i32[1:] - selected.offsets_i32[:-1]
        max_selected_segments = int(row_lengths.max().item()) if row_lengths.numel() else 0
        if (
            resolved_tape_mode == DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE
            and max_selected_segments > DELTA_SMALLRUN16_MAX_SEGMENTS
        ):
            raise ValueError(
                f"{DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE} requires max row segments <= "
                f"{DELTA_SMALLRUN16_MAX_SEGMENTS}, got {max_selected_segments}"
            )
        selected_storage_bytes = _selected_tape_storage_bytes(
            tape_mode=tape_mode,
            selected=selected,
            endpoint_record_edit=endpoint_record_edit,
            endpoint_record_delta_replace=endpoint_record_delta_replace,
            endpoint_record_block_edit=endpoint_record_block_edit,
            coeff_f32=coeff_f32,
            extra_storage_bytes=selected_extra_storage_bytes,
            factorized_coeff_storage_bytes=selected_factorized_coeff_storage_bytes,
            record_bytes_override=(
                _delta_framegroup_record_bytes(resolved_tape_mode)
                if tape_mode == DELTA_AUTO_FRAMEGROUP16_MODE
                else None
            ),
            include_delta_index_storage=not bool(
                endpoint_record_edit_device
                and endpoint_record_edit_device.get("delta_packed_framegroup16_rowdesc_launch_only_fused_mse", False)
            ),
        )
        selected_device = (
            endpoint_record_edit_device
            if tape_mode in endpoint_record_modes
            else (
                _move_endpoint_tape_to_mps(selected)
                if tape_mode in {"endpoint-run", ENDPOINT_RUN_FUSED_MSE_MODE}
                else _move_tape_to_mps(selected, include_mids=tape_mode != OWNER_RUN_FUSED_MSE_NOMID_MODE)
            )
        )
        if tape_mode == ENDPOINT_RUN_FUSED_MSE_MODE:
            selected_device["endpoint_run_fused_mse"] = True
        if tape_mode in {OWNER_RUN_FUSED_MSE_MODE, OWNER_RUN_FUSED_MSE_NOMID_MODE}:
            selected_device["owner_run_fused_mse"] = True
        if tape_mode == OWNER_RUN_FUSED_MSE_NOMID_MODE:
            selected_device["owner_run_fused_mse_nomids"] = True
        selected_segments = int(selected.owners_i32.numel())
        baseline_segments = int(full.owners_i32.numel())
        active_internal_segments = int(active_internal.owners_i32.numel())
        owner_run_segments = int(owner_run.owners_i32.numel())
        endpoint_run_segments = int(endpoint_run.owners_i32.numel())
        baseline_storage_bytes = int(full.storage_bytes)
        active_internal_storage_bytes = int(active_internal.storage_bytes)
        owner_run_storage_bytes = int(owner_run.storage_bytes)
        endpoint_run_storage_bytes = int(endpoint_run.storage_bytes)
        prepared_track_count = int(tape.track_count)
    selected_coeff_storage_bytes = _selected_coeff_storage_bytes(
        tape_mode=tape_mode,
        coeff_f32=coeff_f32,
        factorized_coeff_storage_bytes=selected_factorized_coeff_storage_bytes,
    )
    selected_topology_storage_bytes = max(int(selected_storage_bytes) - int(selected_coeff_storage_bytes), 0)
    selected_schema_storage_by_key = _selected_schema_storage_by_key(
        tape_mode=tape_mode,
        endpoint_record_delta_replace=endpoint_record_delta_replace,
        selected_storage_bytes=selected_storage_bytes,
        selected_coeff_storage_bytes=selected_coeff_storage_bytes,
        extra_storage_bytes=selected_extra_storage_bytes,
        record_bytes_override=(
            _delta_framegroup_record_bytes(resolved_tape_mode)
            if tape_mode == DELTA_AUTO_FRAMEGROUP16_MODE
            else None
        ),
        include_delta_index_storage=not bool(
            endpoint_record_edit_device
            and endpoint_record_edit_device.get("delta_packed_framegroup16_rowdesc_launch_only_fused_mse", False)
        ),
    )
    selected_schema_i16_meta_projection = _selected_schema_i16_meta_projection(
        endpoint_record_delta_replace=endpoint_record_delta_replace,
        selected_schema_storage_by_key=selected_schema_storage_by_key,
        selected_storage_bytes=selected_storage_bytes,
    )
    selected_resident_breakdown = _selected_device_tensor_storage_breakdown(selected_device, device_type="mps")
    selected_resident_coeff_storage_bytes = int(selected_resident_breakdown["coeff_bytes"])
    if tape_mode in OWNER_RUN_DELTA_PACKED_FACTORIZED_MODES:
        resident_by_key = selected_resident_breakdown["by_key"]
        selected_resident_coeff_storage_bytes = int(resident_by_key.get("boundary_f32", 0)) + int(
            resident_by_key.get("track_ray_coeff_f32", 0)
        )
    selected_resident_noncoeff_storage_bytes = max(
        int(selected_resident_breakdown["total_bytes"]) - int(selected_resident_coeff_storage_bytes),
        0,
    )
    return {
        "segment_tape": tape,
        "full": full,
        "active_internal": active_internal,
        "owner_run": owner_run,
        "endpoint_run": endpoint_run,
        "endpoint_record_edit": endpoint_record_edit,
        "endpoint_record_delta_replace": endpoint_record_delta_replace,
        "endpoint_record_block_edit": endpoint_record_block_edit,
        "endpoint_record_delta_render_inputs": endpoint_record_delta_render_inputs,
        "selected": selected,
        "selected_device": selected_device,
        "baseline_segment_metrics_built": not skip_baseline_segment_tapes,
        "experimental_selected_only_owner_run_delta_prep": bool(selected_only_owner_run_delta_prep),
        "experimental_native_owner_run_cutwalk_delta": bool(native_owner_run_cutwalk_delta),
        "full_segments": int(baseline_segments),
        "active_internal_segments": int(active_internal_segments),
        "owner_run_segments": int(owner_run_segments),
        "endpoint_run_segments": int(endpoint_run_segments),
        "endpoint_record_edit_ops": int(endpoint_record_edit.op_type_i32.numel())
        if endpoint_record_edit is not None
        else 0,
        "endpoint_record_edit_change_events": int(endpoint_record_edit.change_frame_i32.numel())
        if endpoint_record_edit is not None
        else 0,
        "endpoint_record_delta_replace_change_events": int(endpoint_record_delta_replace.change_frame_i32.numel())
        if endpoint_record_delta_replace is not None
        else 0,
        "endpoint_record_delta_replace_changed_records": int(endpoint_record_delta_replace.change_owner_i32.numel())
        if endpoint_record_delta_replace is not None
        else 0,
        "endpoint_record_block4_ops": int(endpoint_record_block_edit.op_type_i32.numel())
        if endpoint_record_block_edit is not None
        else 0,
        "endpoint_record_block4_change_events": int(endpoint_record_block_edit.change_frame_i32.numel())
        if endpoint_record_block_edit is not None
        else 0,
        "endpoint_record_block4_changed_records": int(endpoint_record_block_edit.changed_records)
        if endpoint_record_block_edit is not None
        else 0,
        "selected_segments": int(selected_segments),
        "full_storage_bytes": int(baseline_storage_bytes),
        "active_internal_storage_bytes": int(active_internal_storage_bytes),
        "owner_run_storage_bytes": int(owner_run_storage_bytes),
        "endpoint_run_storage_bytes": int(endpoint_run_storage_bytes),
        "endpoint_record_edit_storage_bytes": int(endpoint_record_edit.storage_bytes)
        if endpoint_record_edit is not None
        else 0,
        "endpoint_record_delta_replace_storage_bytes": int(endpoint_record_delta_replace.storage_bytes)
        if endpoint_record_delta_replace is not None
        else 0,
        "endpoint_record_coeff_storage_bytes": int(selected_coeff_storage_bytes),
        "endpoint_record_block4_storage_bytes": int(endpoint_record_block_edit.storage_bytes)
        if endpoint_record_block_edit is not None
        else 0,
        "selected_storage_bytes": selected_storage_bytes,
        "selected_schema_storage_bytes": int(selected_storage_bytes),
        "selected_schema_storage_by_key": dict(selected_schema_storage_by_key),
        "selected_schema_i16_meta_projection_eligible": bool(selected_schema_i16_meta_projection["eligible"]),
        "selected_schema_i16_meta_projected_storage_bytes": int(
            selected_schema_i16_meta_projection["storage_bytes"]
        ),
        "selected_schema_i16_meta_projected_storage_savings_bytes": int(
            selected_schema_i16_meta_projection["savings_bytes"]
        ),
        "selected_schema_i16_meta_projected_storage_by_key": dict(selected_schema_i16_meta_projection["by_key"]),
        "selected_schema_i16_meta_projection_fields": dict(selected_schema_i16_meta_projection["fields"]),
        "selected_topology_storage_bytes": int(selected_topology_storage_bytes),
        "selected_schema_topology_storage_bytes": int(selected_topology_storage_bytes),
        "selected_mps_resident_storage_bytes": int(selected_resident_breakdown["total_bytes"]),
        "selected_mps_resident_noncoeff_storage_bytes": int(selected_resident_noncoeff_storage_bytes),
        "endpoint_record_coeff_mps_resident_storage_bytes": int(selected_resident_coeff_storage_bytes),
        "selected_mps_resident_storage_by_key": dict(selected_resident_breakdown["by_key"]),
        "max_selected_segments_per_sample": int(max_selected_segments),
        "track_count": int(prepared_track_count),
        "tape_mode": tape_mode,
        "tape_mode_resolved": resolved_tape_mode,
        "endpoint_record_source": (
            "gate4-affine"
            if candidate_affine_mode
            else endpoint_record_source
            if tape_mode in endpoint_record_modes
            else "not-used"
        ),
        "gate4_affine_candidate_csr_fused_mse": bool(candidate_affine_mode),
        "gate4_affine_candidate_csr_trackmse_fused_mse": bool(_is_gate4_affine_candidate_trackmse_mode(tape_mode)),
        "gate4_affine_candidate_csr_coeff16_fused_mse": bool(_is_gate4_affine_candidate_coeff16_mode(tape_mode)),
        "gate4_affine_candidate_csr_cap224_fused_mse": bool(_is_gate4_affine_candidate_cap224_mode(tape_mode)),
        "gate4_affine_candidate_csr_densitymask_fused_mse": bool(
            _is_gate4_affine_candidate_densitymask_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_sample_reduce_fused_mse": bool(
            _is_gate4_affine_candidate_sample_reduce_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_sortnet_fused_mse": bool(_is_gate4_affine_candidate_sortnet_mode(tape_mode)),
        "gate4_affine_candidate_csr_framegroup16_cached_fused_mse": bool(
            _is_gate4_affine_candidate_framegroup16_cached_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_sitecache_fused_mse": bool(
            _is_gate4_affine_candidate_sitecache_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_ownerupdate_fused_mse": bool(
            _is_gate4_affine_candidate_ownerupdate_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_ownerupdate_i16_fused_mse": bool(
            _is_gate4_affine_candidate_ownerupdate_i16_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_ownerkeep_fused_mse": bool(
            _is_gate4_affine_candidate_ownerkeep_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_ownerkeep_i16_fused_mse": bool(
            _is_gate4_affine_candidate_ownerkeep_i16_mode(tape_mode)
        ),
        "experimental_native_cut_prep_delta": bool(experimental_native_cut_prep_delta)
        if tape_mode in endpoint_record_modes and endpoint_record_source == "gate4-affine"
        else False,
        "experimental_native_sorted_delta": bool(experimental_native_sorted_delta)
        if tape_mode in endpoint_record_modes and endpoint_record_source == "gate4-affine"
        else False,
        "experimental_cpu_rebase_delta": bool(experimental_cpu_rebase_delta)
        if tape_mode in endpoint_record_modes and endpoint_record_source == "gate4-affine"
        else False,
        "experimental_minimal_packed_delta_device": bool(
            experimental_minimal_packed_delta_device and _delta_mode_uses_packed_framegroup(resolved_tape_mode)
        ),
        "experimental_kernel_order_packed_delta_device": bool(use_kernel_order_packed_delta_device),
        "experimental_launch_only_packed_delta": bool(
            experimental_launch_only_packed_delta and _delta_mode_uses_packed_framegroup(resolved_tape_mode)
        ),
        "experimental_unchecked_launch_only_packed_delta": bool(
            experimental_unchecked_launch_only_packed_delta
            and experimental_launch_only_packed_delta
            and resolved_tape_mode == DELTA_PACKED_FRAMEGROUP16_MODE
        ),
        "experimental_reduce32_launch_only_packed_delta": bool(
            experimental_reduce32_launch_only_packed_delta
            and experimental_launch_only_packed_delta
            and resolved_tape_mode == DELTA_PACKED_FRAMEGROUP16_MODE
        ),
        "experimental_rowselect32_launch_only_packed_delta": bool(
            experimental_rowselect32_launch_only_packed_delta
            and experimental_launch_only_packed_delta
            and resolved_tape_mode == DELTA_PACKED_FRAMEGROUP16_MODE
        ),
        "experimental_rowdesc_launch_only_packed_delta": bool(
            experimental_rowdesc_launch_only_packed_delta
            and experimental_launch_only_packed_delta
            and resolved_tape_mode == DELTA_PACKED_FRAMEGROUP16_MODE
        ),
        "experimental_rowdesc32_launch_only_packed_delta": bool(
            experimental_rowdesc32_launch_only_packed_delta
            and experimental_rowdesc_launch_only_packed_delta
            and experimental_launch_only_packed_delta
            and resolved_tape_mode == DELTA_PACKED_FRAMEGROUP16_MODE
        ),
        "experimental_smallrun16_packed_delta": bool(
            experimental_smallrun16_packed_delta and resolved_tape_mode == DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE
        ),
        "experimental_native_pack_records": bool(experimental_native_pack_records)
        if tape_mode in endpoint_record_modes and endpoint_record_source == "gate4-affine"
        else False,
        "experimental_native_emitted_pack_records": bool(experimental_native_emitted_pack_records)
        if tape_mode in endpoint_record_modes and endpoint_record_source == "gate4-affine"
        else False,
        "experimental_native_emitted_pack_records_effective": bool(effective_native_emitted_pack_records)
        if tape_mode in endpoint_record_modes and endpoint_record_source == "gate4-affine"
        else False,
        "prepare_timings": prepare_timings,
        "gate4_endpoint_metadata": gate4_endpoint_metadata,
    }


def _run_one(
    *,
    config_path: Path,
    frame_count: int,
    render_size: int,
    site_count: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    synthetic_motion: SyntheticRayMotion,
    steps: int,
    warmup_steps: int,
    lr: float,
    beta1: float,
    beta2: float,
    adam_eps: float,
    optimizer_mode: str,
    segment_tape_vjp_mode: str,
    tape_mode: str,
    site_initialization: str = SITE_INITIALIZATION_LEGACY_SPARSE,
    edit_block_size: int = 4,
    allow_repeat_loaded_frames: bool = False,
    endpoint_record_source: str = "slow-owner-run",
    gate4_time_slabs: int = 1,
    gate4_residual_depth_padding: float = 0.001,
    experimental_native_cut_prep_delta: bool = False,
    experimental_native_sorted_delta: bool = False,
    experimental_minimal_packed_delta_device: bool = False,
    experimental_cpu_rebase_delta: bool = False,
    experimental_kernel_order_packed_delta_device: bool = False,
    experimental_smallrun16_packed_delta: bool = False,
    experimental_launch_only_packed_delta: bool = False,
    experimental_unchecked_launch_only_packed_delta: bool = False,
    experimental_reduce32_launch_only_packed_delta: bool = False,
    experimental_rowselect32_launch_only_packed_delta: bool = False,
    experimental_rowdesc_launch_only_packed_delta: bool = False,
    experimental_rowdesc32_launch_only_packed_delta: bool = False,
    experimental_native_pack_records: bool = False,
    experimental_native_emitted_pack_records: bool = False,
    experimental_selected_only_owner_run_delta_prep: bool = False,
    experimental_native_owner_run_cutwalk_delta: bool = False,
    defer_heldout_device: bool = False,
) -> dict[str, Any]:
    if optimizer_mode not in {"manual-vjp", "autograd"}:
        raise ValueError("optimizer_mode must be 'manual-vjp' or 'autograd'")
    if tape_mode in {
        "endpoint-record-edit-block-coeff16",
        "endpoint-record-edit-block-coeff16-fused-mse",
        "endpoint-record-edit-block-coeff16-packed-fused-mse",
        "endpoint-record-edit-block-coeff16-i16-fused-mse",
        "endpoint-record-edit-block-coeff16-i16x3-fused-mse",
    } and optimizer_mode != "manual-vjp":
        raise ValueError(f"{tape_mode} currently supports only optimizer-mode=manual-vjp")
    if tape_mode in {
        OWNER_RUN_FUSED_MSE_MODE,
        OWNER_RUN_FUSED_MSE_NOMID_MODE,
        ENDPOINT_RUN_FUSED_MSE_MODE,
        "endpoint-record-edit-fused-mse",
        "endpoint-record-edit-coeff16-fused-mse",
        "endpoint-record-delta-replace-coeff16-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x3-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse",
        DELTA_I16X3_FRAMEGROUP16_MODE,
        DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
        DELTA_PACKED_SCALAR_MODE,
        DELTA_PACKED_FRAMEGROUP16_MODE,
        DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
        DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
        DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
        OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
        DELTA_AUTO_FRAMEGROUP16_MODE,
        "endpoint-record-delta-replace-coeff16-i16x4-fused-mse",
        "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
        "endpoint-record-edit-block-coeff-fused-mse",
        "endpoint-record-edit-block-coeff16-fused-mse",
        "endpoint-record-edit-block-coeff16-packed-fused-mse",
        "endpoint-record-edit-block-coeff16-i16-fused-mse",
        "endpoint-record-edit-block-coeff16-i16x3-fused-mse",
        *GATE4_AFFINE_CANDIDATE_FUSED_MSE_MODES,
    }:
        if optimizer_mode != "manual-vjp":
            raise ValueError(f"{tape_mode} currently supports only optimizer-mode=manual-vjp")
    if segment_tape_vjp_mode not in {"direct_atomic_grad_only", "direct_atomic_track"}:
        raise ValueError("segment_tape_vjp_mode must be 'direct_atomic_grad_only' or 'direct_atomic_track'")
    if tape_mode in ENDPOINT_SEMANTIC_TAPE_MODES and segment_tape_vjp_mode != "direct_atomic_grad_only":
        raise ValueError(f"{tape_mode} tape mode currently supports only direct_atomic_grad_only VJP")
    cfg = _load_config(config_path, max_frames=frame_count, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    train_rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    train_frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    heldout_targets = data["heldout_targets"]
    heldout_rays = data["heldout_rays"]
    heldout_frame_indices = data["heldout_frame_indices"]
    if heldout_targets is None or heldout_rays is None or heldout_frame_indices is None:
        raise ValueError("owner-run train/eval requires heldout targets, rays, and frame indices")
    heldout_targets = heldout_targets.detach().cpu().to(dtype=torch.float32)
    heldout_rays = heldout_rays.detach().cpu().to(dtype=torch.float32)
    heldout_frame_indices = heldout_frame_indices.detach().cpu().to(dtype=torch.long)
    loaded_frame_count = int(data["frame_count"])
    targets, train_rays, train_frame_indices, train_repeated = _fit_loaded_frame_count(
        split_name="train",
        targets=targets,
        rays=train_rays,
        frame_indices=train_frame_indices,
        loaded_frame_count=loaded_frame_count,
        requested_frame_count=frame_count,
        allow_repeat_loaded_frames=allow_repeat_loaded_frames,
    )
    heldout_targets, heldout_rays, heldout_frame_indices, heldout_repeated = _fit_loaded_frame_count(
        split_name="heldout",
        targets=heldout_targets,
        rays=heldout_rays,
        frame_indices=heldout_frame_indices,
        loaded_frame_count=loaded_frame_count,
        requested_frame_count=frame_count,
        allow_repeat_loaded_frames=allow_repeat_loaded_frames,
    )
    repeated_loaded_frames = bool(train_repeated or heldout_repeated)

    train_rays = apply_synthetic_ray_motion(
        train_rays,
        frame_indices=train_frame_indices,
        frame_count=frame_count,
        motion=synthetic_motion,
    )
    heldout_rays = apply_synthetic_ray_motion(
        heldout_rays,
        frame_indices=heldout_frame_indices,
        frame_count=frame_count,
        motion=synthetic_motion,
    )
    sites = initialize_sites_from_train_samples(
        targets=targets,
        rays=train_rays,
        frame_indices=train_frame_indices,
        frame_count=frame_count,
        site_count=site_count,
        near=near,
        far=far,
        density=density,
        initialization=site_initialization,
    )
    site_rgba_initial_cpu = torch.tensor([site.rgba for site in sites], dtype=torch.float32)
    train_tape = _prepare_owner_run_tapes(
        sites=sites,
        rays=train_rays,
        frame_indices=train_frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
        site_rgba=site_rgba_initial_cpu,
        tape_mode=tape_mode,
        edit_block_size=edit_block_size,
        endpoint_record_source=endpoint_record_source,
        gate4_time_slabs=gate4_time_slabs,
        gate4_residual_depth_padding=gate4_residual_depth_padding,
        experimental_native_cut_prep_delta=experimental_native_cut_prep_delta,
        experimental_native_sorted_delta=experimental_native_sorted_delta,
        experimental_minimal_packed_delta_device=experimental_minimal_packed_delta_device,
        experimental_cpu_rebase_delta=experimental_cpu_rebase_delta,
        experimental_kernel_order_packed_delta_device=experimental_kernel_order_packed_delta_device,
        experimental_smallrun16_packed_delta=experimental_smallrun16_packed_delta,
        experimental_launch_only_packed_delta=experimental_launch_only_packed_delta,
        experimental_unchecked_launch_only_packed_delta=experimental_unchecked_launch_only_packed_delta,
        experimental_reduce32_launch_only_packed_delta=experimental_reduce32_launch_only_packed_delta,
        experimental_rowselect32_launch_only_packed_delta=experimental_rowselect32_launch_only_packed_delta,
        experimental_rowdesc_launch_only_packed_delta=experimental_rowdesc_launch_only_packed_delta,
        experimental_rowdesc32_launch_only_packed_delta=experimental_rowdesc32_launch_only_packed_delta,
        experimental_native_pack_records=experimental_native_pack_records,
        experimental_native_emitted_pack_records=experimental_native_emitted_pack_records,
        experimental_selected_only_owner_run_delta_prep=experimental_selected_only_owner_run_delta_prep,
        experimental_native_owner_run_cutwalk_delta=experimental_native_owner_run_cutwalk_delta,
    )

    def _prepare_heldout_tape() -> dict[str, Any]:
        return _prepare_owner_run_tapes(
            sites=sites,
            rays=heldout_rays,
            frame_indices=heldout_frame_indices,
            frame_count=frame_count,
            near=near,
            far=far,
            invalid_epsilon=invalid_epsilon,
            transmittance_threshold=transmittance_threshold,
            site_rgba=site_rgba_initial_cpu,
            tape_mode=tape_mode,
            edit_block_size=edit_block_size,
            endpoint_record_source=endpoint_record_source,
            gate4_time_slabs=gate4_time_slabs,
            gate4_residual_depth_padding=gate4_residual_depth_padding,
            experimental_native_cut_prep_delta=experimental_native_cut_prep_delta,
            experimental_native_sorted_delta=experimental_native_sorted_delta,
            experimental_minimal_packed_delta_device=experimental_minimal_packed_delta_device,
            experimental_cpu_rebase_delta=experimental_cpu_rebase_delta,
            experimental_kernel_order_packed_delta_device=experimental_kernel_order_packed_delta_device,
            experimental_smallrun16_packed_delta=experimental_smallrun16_packed_delta,
            experimental_launch_only_packed_delta=experimental_launch_only_packed_delta,
            experimental_unchecked_launch_only_packed_delta=experimental_unchecked_launch_only_packed_delta,
            experimental_reduce32_launch_only_packed_delta=experimental_reduce32_launch_only_packed_delta,
            experimental_rowselect32_launch_only_packed_delta=experimental_rowselect32_launch_only_packed_delta,
            experimental_rowdesc_launch_only_packed_delta=experimental_rowdesc_launch_only_packed_delta,
            experimental_rowdesc32_launch_only_packed_delta=experimental_rowdesc32_launch_only_packed_delta,
            experimental_native_pack_records=experimental_native_pack_records,
            experimental_native_emitted_pack_records=experimental_native_emitted_pack_records,
            experimental_selected_only_owner_run_delta_prep=experimental_selected_only_owner_run_delta_prep,
            experimental_native_owner_run_cutwalk_delta=experimental_native_owner_run_cutwalk_delta,
        )

    heldout_tape = None if defer_heldout_device else _prepare_heldout_tape()
    device = torch.device("mps")
    train_targets = targets.to(device)
    heldout_targets_device = None if defer_heldout_device else heldout_targets.to(device)
    site_rgba = site_rgba_initial_cpu.to(device=device).contiguous()
    if optimizer_mode == "autograd":
        site_rgba = site_rgba.detach().clone().requires_grad_(True)
    site_rgba_initial = site_rgba.detach().clone()
    exp_avg = torch.zeros_like(site_rgba)
    exp_avg_sq = torch.zeros_like(site_rgba)
    optimizer = (
        torch.optim.Adam([site_rgba], lr=lr, betas=(beta1, beta2), eps=adam_eps)
        if optimizer_mode == "autograd"
        else None
    )
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    train_sample_count, train_height, train_width, _payload = train_rays.shape
    heldout_sample_count, heldout_height, heldout_width, _heldout_payload = heldout_rays.shape
    train_view_count = int(train_sample_count // frame_count)
    heldout_view_count = int(heldout_sample_count // frame_count)
    train_track_count = int(train_tape["track_count"])
    heldout_track_count = int(heldout_tape["track_count"]) if heldout_tape is not None else 0
    fused_mse_mode = bool(
        train_tape["selected_device"].get("edit_fused_mse", False)
        or train_tape["selected_device"].get("delta_replace_fused_mse", False)
        or train_tape["selected_device"].get("block_coeff_fused_mse", False)
        or train_tape["selected_device"].get("affine_candidate_fused_mse", False)
        or train_tape["selected_device"].get("endpoint_run_fused_mse", False)
        or train_tape["selected_device"].get("owner_run_fused_mse", False)
    )
    train_targets_track = (
        _track_major_rgb_from_image(
            train_targets,
            view_count=train_view_count,
            frame_count=frame_count,
            height=int(train_height),
            width=int(train_width),
        )
        if fused_mse_mode
        else None
    )
    framegroup_fused_mse_objective = None
    if (
        fused_mse_mode
        and optimizer_mode == "autograd"
        and bool(train_tape["selected_device"].get("delta_i16x3_framegroup16_fused_mse", False))
    ):
        framegroup_fused_mse_objective = _delta_replace_coeff16_framegroup_fused_mse_objective(
            tape_device=train_tape["selected_device"],
            op_config=op_config,
            track_count=train_track_count,
            frame_count=frame_count,
        )
    world_foam_objective_adapter = _world_foam_objective_adapter_metadata(framegroup_fused_mse_objective)

    step_rows: list[dict[str, float]] = []
    loss_history: list[float] = []
    first_grad_abs_sum = 0.0
    total_steps = warmup_steps + steps
    for step in range(total_steps):
        step_start = time.perf_counter()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        fused_loss_vjp_s = 0.0
        if fused_mse_mode:
            if train_targets_track is None:
                raise RuntimeError("fused MSE mode did not build track-major targets")
            fused_start = time.perf_counter()
            if bool(train_tape["selected_device"].get("affine_candidate_fused_mse", False)):
                loss, grad_site_rgba = _affine_candidate_fused_mse_loss_vjp(
                    tape_device=train_tape["selected_device"],
                    site_rgba=site_rgba,
                    target_rgb_track=train_targets_track,
                    op_config=op_config,
                )
            elif bool(train_tape["selected_device"].get("endpoint_run_fused_mse", False)):
                loss, grad_site_rgba = _endpoint_run_fused_mse_loss_vjp(
                    tape_device=train_tape["selected_device"],
                    site_rgba=site_rgba,
                    target_rgb_track=train_targets_track,
                    op_config=op_config,
                    track_count=train_track_count,
                    frame_count=frame_count,
                )
            elif bool(train_tape["selected_device"].get("owner_run_fused_mse", False)):
                loss, grad_site_rgba = _segment_tape_fused_mse_loss_vjp(
                    tape_device=train_tape["selected_device"],
                    site_rgba=site_rgba,
                    target_rgb_track=train_targets_track,
                    op_config=op_config,
                    track_count=train_track_count,
                    frame_count=frame_count,
                )
            elif bool(train_tape["selected_device"].get("edit_fused_mse", False)):
                loss, grad_site_rgba = _edit_fused_mse_loss_vjp(
                    tape_device=train_tape["selected_device"],
                    site_rgba=site_rgba,
                    target_rgb_track=train_targets_track,
                    op_config=op_config,
                    track_count=train_track_count,
                    frame_count=frame_count,
                )
            elif bool(train_tape["selected_device"].get("delta_replace_fused_mse", False)):
                if optimizer_mode == "autograd":
                    if framegroup_fused_mse_objective is None:
                        raise RuntimeError("only the delta i16x3 framegroup fused-MSE path has an autograd wrapper")
                    loss = framegroup_fused_mse_objective.loss(
                        site_rgba=site_rgba,
                        target_rgb=train_targets_track,
                    )
                    loss.backward()
                    if site_rgba.grad is None:
                        raise RuntimeError("framegroup fused MSE autograd did not produce site_rgba.grad")
                    grad_site_rgba = site_rgba.grad.detach()
                else:
                    loss, grad_site_rgba = _delta_replace_coeff16_fused_mse_loss_vjp(
                        tape_device=train_tape["selected_device"],
                        site_rgba=site_rgba,
                        target_rgb_track=train_targets_track,
                        op_config=op_config,
                        track_count=train_track_count,
                        frame_count=frame_count,
                    )
            else:
                loss, grad_site_rgba = _block_coeff_fused_mse_loss_vjp(
                    tape_device=train_tape["selected_device"],
                    site_rgba=site_rgba,
                    target_rgb_track=train_targets_track,
                    op_config=op_config,
                    track_count=train_track_count,
                    frame_count=frame_count,
                )
            torch.mps.synchronize()
            fused_loss_vjp_s = time.perf_counter() - fused_start
            render_s = 0.0
            loss_s = 0.0
            backward_s = fused_loss_vjp_s
        else:
            render_start = time.perf_counter()
            rendered = _render_owner_run_rgb(
                tape_device=train_tape["selected_device"],
                site_rgba=site_rgba,
                op_config=op_config,
                track_count=train_track_count,
                frame_count=frame_count,
                view_count=train_view_count,
                height=int(train_height),
                width=int(train_width),
                autograd_vjp_mode=segment_tape_vjp_mode if optimizer_mode == "autograd" else None,
            )
            torch.mps.synchronize()
            render_s = time.perf_counter() - render_start
            loss_start = time.perf_counter()
            diff = rendered - train_targets
            loss = diff.square().mean()
            grad_rgb_image = (2.0 / float(diff.numel())) * diff if optimizer_mode == "manual-vjp" else None
            torch.mps.synchronize()
            loss_s = time.perf_counter() - loss_start
            backward_start = time.perf_counter()
            if optimizer_mode == "manual-vjp":
                if grad_rgb_image is None:
                    raise RuntimeError("manual VJP mode did not build an RGB gradient seed")
                grad_site_rgba = _owner_run_vjp_rgb_only(
                    tape_device=train_tape["selected_device"],
                    site_rgba=site_rgba,
                    grad_rgb_image=grad_rgb_image.contiguous(),
                    op_config=op_config,
                    track_count=train_track_count,
                    frame_count=frame_count,
                    view_count=train_view_count,
                    height=int(train_height),
                    width=int(train_width),
                )
            else:
                loss.backward()
                if site_rgba.grad is None:
                    raise RuntimeError("segment tape autograd did not produce site_rgba.grad")
                grad_site_rgba = site_rgba.grad.detach()
            torch.mps.synchronize()
            backward_s = time.perf_counter() - backward_start
        if step == 0:
            first_grad_abs_sum = float(grad_site_rgba.detach().abs().sum().cpu().item())
        optimizer_start = time.perf_counter()
        if optimizer is None:
            _adam_update(
                param=site_rgba,
                grad=grad_site_rgba,
                exp_avg=exp_avg,
                exp_avg_sq=exp_avg_sq,
                step_index=step + 1,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=adam_eps,
            )
        else:
            optimizer.step()
            with torch.no_grad():
                site_rgba[:, :3].clamp_(0.0, 1.0)
                site_rgba[:, 3].clamp_(min=0.01)
        torch.mps.synchronize()
        optimizer_s = time.perf_counter() - optimizer_start
        row = {
            "render": float(render_s),
            "loss_eval": float(loss_s),
            "backward": float(backward_s),
            "optimizer": float(optimizer_s),
            "total": float(time.perf_counter() - step_start),
            "loss": float(loss.detach().cpu().item()),
        }
        if fused_mse_mode:
            row["fused_loss_vjp"] = float(fused_loss_vjp_s)
        if step >= warmup_steps:
            step_rows.append(row)
            loss_history.append(row["loss"])

    heldout_tape_prepared_after_timing = False
    if heldout_tape is None:
        heldout_tape = _prepare_heldout_tape()
        heldout_track_count = int(heldout_tape["track_count"])
        heldout_tape_prepared_after_timing = True
    if heldout_targets_device is None:
        heldout_targets_device = heldout_targets.to(device)

    with torch.no_grad():
        final_train = _render_owner_run_rgb(
            tape_device=_render_device_for_tape(train_tape),
            site_rgba=site_rgba,
            op_config=op_config,
            track_count=train_track_count,
            frame_count=frame_count,
            view_count=train_view_count,
            height=int(train_height),
            width=int(train_width),
        )
        final_heldout = _render_owner_run_rgb(
            tape_device=_render_device_for_tape(heldout_tape),
            site_rgba=site_rgba,
            op_config=op_config,
            track_count=heldout_track_count,
            frame_count=frame_count,
            view_count=heldout_view_count,
            height=int(heldout_height),
            width=int(heldout_width),
        )
        torch.mps.synchronize()

    train_metrics = _metrics(final_train, train_targets)
    heldout_metrics = _metrics(final_heldout, heldout_targets_device)
    param_update = float((site_rgba.detach() - site_rgba_initial).abs().max().cpu().item())
    selected_vjp_row_cap = (
        256
        if _is_gate4_affine_candidate_fused_mse_mode(tape_mode)
        else 129
    )
    acceptance = {
        "loss_decreased": bool(loss_history and train_metrics["mse"] < loss_history[0]),
        "gradients_nonzero": first_grad_abs_sum > 0.0,
        "parameters_updated": param_update > 1.0e-6,
        "selected_tape_segments_below_full": int(train_tape["selected_segments"]) <= int(train_tape["full_segments"]),
        "owner_run_segments_below_full": int(train_tape["selected_segments"]) <= int(train_tape["full_segments"]),
        "selected_tape_vjp_under_segment_cap": int(train_tape["max_selected_segments_per_sample"])
        <= selected_vjp_row_cap
        and int(heldout_tape["max_selected_segments_per_sample"]) <= selected_vjp_row_cap,
        "owner_run_vjp_under_segment_cap": int(train_tape["max_selected_segments_per_sample"])
        <= selected_vjp_row_cap
        and int(heldout_tape["max_selected_segments_per_sample"]) <= selected_vjp_row_cap,
        "outputs_are_finite": bool(torch.isfinite(final_train).all().item() and torch.isfinite(final_heldout).all().item()),
    }
    return {
        "frame_count": frame_count,
        "loaded_frame_count": loaded_frame_count,
        "repeat_loaded_frames": repeated_loaded_frames,
        "repeat_loaded_frames_scope": (
            "synthetic repeated-fixture speed-scaling smoke"
            if repeated_loaded_frames
            else "real loaded frame count"
        ),
        "render_size": render_size,
        "site_count": site_count,
        "site_initialization": site_initialization,
        "tape_mode": tape_mode,
        "tape_mode_resolved": str(train_tape.get("tape_mode_resolved", tape_mode)),
        "endpoint_record_source": str(train_tape.get("endpoint_record_source", "not-used")),
        "gate4_affine_candidate_csr_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_fused_mse", False)
        ),
        "gate4_affine_candidate_csr_trackmse_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_trackmse_fused_mse", False)
        ),
        "gate4_affine_candidate_csr_coeff16_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_coeff16_fused_mse", False)
        ),
        "gate4_affine_candidate_csr_cap224_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_cap224_fused_mse", False)
        ),
        "gate4_affine_candidate_csr_densitymask_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_densitymask_fused_mse", False)
        ),
        "gate4_affine_candidate_csr_sample_reduce_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_sample_reduce_fused_mse", False)
        ),
        "gate4_affine_candidate_csr_sortnet_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_sortnet_fused_mse", False)
        ),
        "gate4_affine_candidate_csr_framegroup16_cached_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_framegroup16_cached_fused_mse", False)
        ),
        "gate4_affine_candidate_csr_sitecache_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_sitecache_fused_mse", False)
        ),
        "gate4_affine_candidate_csr_ownerupdate_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_ownerupdate_fused_mse", False)
        ),
        "gate4_affine_candidate_csr_ownerupdate_i16_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_ownerupdate_i16_fused_mse", False)
        ),
        "gate4_affine_candidate_csr_ownerkeep_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_ownerkeep_fused_mse", False)
        ),
        "gate4_affine_candidate_csr_ownerkeep_i16_fused_mse": bool(
            train_tape.get("gate4_affine_candidate_csr_ownerkeep_i16_fused_mse", False)
        ),
        "experimental_native_cut_prep_delta": bool(train_tape.get("experimental_native_cut_prep_delta", False)),
        "experimental_native_sorted_delta": bool(train_tape.get("experimental_native_sorted_delta", False)),
        "experimental_cpu_rebase_delta": bool(train_tape.get("experimental_cpu_rebase_delta", False)),
        "experimental_minimal_packed_delta_device": bool(
            train_tape.get("experimental_minimal_packed_delta_device", False)
        ),
        "experimental_kernel_order_packed_delta_device": bool(
            train_tape.get("experimental_kernel_order_packed_delta_device", False)
        ),
        "experimental_launch_only_packed_delta": bool(
            train_tape.get("experimental_launch_only_packed_delta", False)
        ),
        "experimental_unchecked_launch_only_packed_delta": bool(
            train_tape.get("experimental_unchecked_launch_only_packed_delta", False)
        ),
        "experimental_reduce32_launch_only_packed_delta": bool(
            train_tape.get("experimental_reduce32_launch_only_packed_delta", False)
        ),
        "experimental_rowselect32_launch_only_packed_delta": bool(
            train_tape.get("experimental_rowselect32_launch_only_packed_delta", False)
        ),
        "experimental_rowdesc_launch_only_packed_delta": bool(
            train_tape.get("experimental_rowdesc_launch_only_packed_delta", False)
        ),
        "experimental_rowdesc32_launch_only_packed_delta": bool(
            train_tape.get("experimental_rowdesc32_launch_only_packed_delta", False)
        ),
        "experimental_smallrun16_packed_delta": bool(
            train_tape.get("experimental_smallrun16_packed_delta", False)
        ),
        "experimental_native_pack_records": bool(train_tape.get("experimental_native_pack_records", False)),
        "experimental_native_emitted_pack_records": bool(
            train_tape.get("experimental_native_emitted_pack_records", False)
        ),
        "experimental_native_emitted_pack_records_effective": bool(
            train_tape.get("experimental_native_emitted_pack_records_effective", False)
        ),
        "experimental_selected_only_owner_run_delta_prep": bool(
            train_tape.get("experimental_selected_only_owner_run_delta_prep", False)
        ),
        "experimental_native_owner_run_cutwalk_delta": bool(
            train_tape.get("experimental_native_owner_run_cutwalk_delta", False)
        ),
        "defer_heldout_device": bool(defer_heldout_device),
        "heldout_tape_prepared_after_timing": bool(heldout_tape_prepared_after_timing),
        "timed_mps_residency_scope": (
            "train_tape_targets_site_only" if defer_heldout_device else "train_and_heldout_tapes_targets_site"
        ),
        "gate4_endpoint_train_metadata": train_tape["gate4_endpoint_metadata"],
        "gate4_endpoint_heldout_metadata": heldout_tape["gate4_endpoint_metadata"],
        "prepare_timings": {
            "train": train_tape["prepare_timings"],
            "heldout": heldout_tape["prepare_timings"],
        },
        "edit_block_size": int(edit_block_size),
        "steps": steps,
        "warmup_steps": warmup_steps,
        "lr": lr,
        "optimizer_mode": optimizer_mode,
        "segment_tape_vjp_mode": segment_tape_vjp_mode if optimizer_mode == "autograd" else "manual_direct_atomic_grad_only",
        "selected_tape_vjp_row_cap": int(selected_vjp_row_cap),
        "initial_measured_loss": loss_history[0] if loss_history else None,
        "final_train_mse": train_metrics["mse"],
        "final_train_psnr": train_metrics["psnr"],
        "final_train_l1": train_metrics["l1"],
        "final_heldout_mse": heldout_metrics["mse"],
        "final_heldout_psnr": heldout_metrics["psnr"],
        "final_heldout_l1": heldout_metrics["l1"],
        "step_summary": _summarize_steps(step_rows),
        "first_grad_abs_sum": first_grad_abs_sum,
        "parameter_update_abs_max": param_update,
        "train_full_segments": int(train_tape["full_segments"]),
        "train_active_internal_segments": int(train_tape["active_internal_segments"]),
        "train_owner_run_segments": int(train_tape["owner_run_segments"]),
        "train_endpoint_run_segments": int(train_tape["endpoint_run_segments"]),
        "train_endpoint_record_edit_ops": int(train_tape["endpoint_record_edit_ops"]),
        "train_endpoint_record_edit_change_events": int(train_tape["endpoint_record_edit_change_events"]),
        "train_endpoint_record_delta_replace_change_events": int(
            train_tape["endpoint_record_delta_replace_change_events"]
        ),
        "train_endpoint_record_delta_replace_changed_records": int(
            train_tape["endpoint_record_delta_replace_changed_records"]
        ),
        "train_endpoint_record_block4_ops": int(train_tape["endpoint_record_block4_ops"]),
        "train_endpoint_record_block4_change_events": int(train_tape["endpoint_record_block4_change_events"]),
        "train_endpoint_record_block4_changed_records": int(train_tape["endpoint_record_block4_changed_records"]),
        "train_selected_tape_segments": int(train_tape["selected_segments"]),
        "train_owner_run_segments_vs_full": float(train_tape["owner_run_segments"]) / float(train_tape["full_segments"]),
        "train_endpoint_run_segments_vs_full": float(train_tape["endpoint_run_segments"])
        / float(train_tape["full_segments"]),
        "train_selected_tape_segments_vs_full": float(train_tape["selected_segments"])
        / float(train_tape["full_segments"]),
        "train_full_storage_bytes": int(train_tape["full_storage_bytes"]),
        "train_active_internal_storage_bytes": int(train_tape["active_internal_storage_bytes"]),
        "train_owner_run_storage_bytes": int(train_tape["owner_run_storage_bytes"]),
        "train_endpoint_run_storage_bytes": int(train_tape["endpoint_run_storage_bytes"]),
        "train_endpoint_record_edit_storage_bytes": int(train_tape["endpoint_record_edit_storage_bytes"]),
        "train_endpoint_record_delta_replace_storage_bytes": int(
            train_tape["endpoint_record_delta_replace_storage_bytes"]
        ),
        "train_endpoint_record_coeff_storage_bytes": int(train_tape["endpoint_record_coeff_storage_bytes"]),
        "train_endpoint_record_block4_storage_bytes": int(train_tape["endpoint_record_block4_storage_bytes"]),
        "train_selected_tape_storage_bytes": int(train_tape["selected_storage_bytes"]),
        "train_selected_tape_schema_storage_bytes": int(train_tape["selected_schema_storage_bytes"]),
        "train_selected_tape_schema_storage_by_key": dict(train_tape["selected_schema_storage_by_key"]),
        "train_selected_tape_schema_i16_meta_projection_eligible": bool(
            train_tape["selected_schema_i16_meta_projection_eligible"]
        ),
        "train_selected_tape_schema_i16_meta_projected_storage_bytes": int(
            train_tape["selected_schema_i16_meta_projected_storage_bytes"]
        ),
        "train_selected_tape_schema_i16_meta_projected_storage_savings_bytes": int(
            train_tape["selected_schema_i16_meta_projected_storage_savings_bytes"]
        ),
        "train_selected_tape_schema_i16_meta_projected_storage_by_key": dict(
            train_tape["selected_schema_i16_meta_projected_storage_by_key"]
        ),
        "train_selected_tape_schema_i16_meta_projection_fields": dict(
            train_tape["selected_schema_i16_meta_projection_fields"]
        ),
        "train_selected_tape_topology_storage_bytes": int(train_tape["selected_topology_storage_bytes"]),
        "train_selected_tape_schema_topology_storage_bytes": int(train_tape["selected_schema_topology_storage_bytes"]),
        "train_selected_tape_mps_resident_storage_bytes": int(train_tape["selected_mps_resident_storage_bytes"]),
        "train_selected_tape_mps_resident_noncoeff_storage_bytes": int(
            train_tape["selected_mps_resident_noncoeff_storage_bytes"]
        ),
        "train_selected_tape_mps_resident_storage_by_key": dict(
            train_tape["selected_mps_resident_storage_by_key"]
        ),
        "train_endpoint_record_coeff_mps_resident_storage_bytes": int(
            train_tape["endpoint_record_coeff_mps_resident_storage_bytes"]
        ),
        "train_baseline_segment_metrics_built": bool(train_tape["baseline_segment_metrics_built"]),
        "train_owner_run_storage_vs_full": float(train_tape["owner_run_storage_bytes"])
        / float(train_tape["full_storage_bytes"]),
        "train_endpoint_run_storage_vs_full": float(train_tape["endpoint_run_storage_bytes"])
        / float(train_tape["full_storage_bytes"]),
        "train_endpoint_record_edit_storage_vs_full": float(train_tape["endpoint_record_edit_storage_bytes"])
        / float(train_tape["full_storage_bytes"])
        if int(train_tape["endpoint_record_edit_storage_bytes"]) > 0
        else 0.0,
        "train_endpoint_record_edit_storage_vs_endpoint_run": float(train_tape["endpoint_record_edit_storage_bytes"])
        / float(train_tape["endpoint_run_storage_bytes"])
        if int(train_tape["endpoint_record_edit_storage_bytes"]) > 0
        else 0.0,
        "train_endpoint_record_delta_replace_storage_vs_full": float(
            train_tape["endpoint_record_delta_replace_storage_bytes"]
        )
        / float(train_tape["full_storage_bytes"])
        if int(train_tape["endpoint_record_delta_replace_storage_bytes"]) > 0
        else 0.0,
        "train_endpoint_record_delta_replace_storage_vs_endpoint_run": float(
            train_tape["endpoint_record_delta_replace_storage_bytes"]
        )
        / float(train_tape["endpoint_run_storage_bytes"])
        if int(train_tape["endpoint_record_delta_replace_storage_bytes"]) > 0
        else 0.0,
        "train_endpoint_record_coeff_storage_vs_selected": float(
            train_tape["endpoint_record_coeff_storage_bytes"]
        )
        / float(train_tape["selected_storage_bytes"])
        if int(train_tape["endpoint_record_coeff_storage_bytes"]) > 0
        else 0.0,
        "train_endpoint_record_coeff_mps_resident_storage_vs_selected_mps_resident": float(
            train_tape["endpoint_record_coeff_mps_resident_storage_bytes"]
        )
        / float(train_tape["selected_mps_resident_storage_bytes"])
        if int(train_tape["endpoint_record_coeff_mps_resident_storage_bytes"]) > 0
        and int(train_tape["selected_mps_resident_storage_bytes"]) > 0
        else 0.0,
        "train_selected_tape_topology_storage_vs_full": float(train_tape["selected_topology_storage_bytes"])
        / float(train_tape["full_storage_bytes"]),
        "train_endpoint_record_block4_storage_vs_full": float(train_tape["endpoint_record_block4_storage_bytes"])
        / float(train_tape["full_storage_bytes"])
        if int(train_tape["endpoint_record_block4_storage_bytes"]) > 0
        else 0.0,
        "train_endpoint_record_block4_storage_vs_endpoint_run": float(
            train_tape["endpoint_record_block4_storage_bytes"]
        )
        / float(train_tape["endpoint_run_storage_bytes"])
        if int(train_tape["endpoint_record_block4_storage_bytes"]) > 0
        else 0.0,
        "train_selected_tape_storage_vs_full": float(train_tape["selected_storage_bytes"])
        / float(train_tape["full_storage_bytes"]),
        "heldout_full_segments": int(heldout_tape["full_segments"]),
        "heldout_active_internal_segments": int(heldout_tape["active_internal_segments"]),
        "heldout_owner_run_segments": int(heldout_tape["owner_run_segments"]),
        "heldout_endpoint_run_segments": int(heldout_tape["endpoint_run_segments"]),
        "heldout_endpoint_record_edit_ops": int(heldout_tape["endpoint_record_edit_ops"]),
        "heldout_endpoint_record_edit_change_events": int(heldout_tape["endpoint_record_edit_change_events"]),
        "heldout_endpoint_record_delta_replace_change_events": int(
            heldout_tape["endpoint_record_delta_replace_change_events"]
        ),
        "heldout_endpoint_record_delta_replace_changed_records": int(
            heldout_tape["endpoint_record_delta_replace_changed_records"]
        ),
        "heldout_endpoint_record_block4_ops": int(heldout_tape["endpoint_record_block4_ops"]),
        "heldout_endpoint_record_block4_change_events": int(heldout_tape["endpoint_record_block4_change_events"]),
        "heldout_endpoint_record_block4_changed_records": int(heldout_tape["endpoint_record_block4_changed_records"]),
        "heldout_selected_tape_segments": int(heldout_tape["selected_segments"]),
        "heldout_baseline_segment_metrics_built": bool(heldout_tape["baseline_segment_metrics_built"]),
        "heldout_owner_run_segments_vs_full": float(heldout_tape["owner_run_segments"])
        / float(heldout_tape["full_segments"]),
        "heldout_endpoint_run_segments_vs_full": float(heldout_tape["endpoint_run_segments"])
        / float(heldout_tape["full_segments"]),
        "heldout_selected_tape_segments_vs_full": float(heldout_tape["selected_segments"])
        / float(heldout_tape["full_segments"]),
        "max_selected_tape_segments_per_sample": max(
            int(train_tape["max_selected_segments_per_sample"]),
            int(heldout_tape["max_selected_segments_per_sample"]),
        ),
        "max_owner_run_segments_per_sample": max(
            int(train_tape["max_selected_segments_per_sample"]),
            int(heldout_tape["max_selected_segments_per_sample"]),
        ),
        "gradient_scope": f"fixed_geometry_{tape_mode}_rgb_only_site_rgba_{optimizer_mode}",
        "world_foam_objective_adapter": world_foam_objective_adapter,
        "acceptance": acceptance,
        "status": "ok" if all(acceptance.values()) else "failed",
    }


def run_train_eval(
    *,
    config_path: Path,
    frame_counts: tuple[int, ...],
    render_size: int,
    site_count: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    synthetic_motion: SyntheticRayMotion,
    steps: int,
    warmup_steps: int,
    lr: float,
    beta1: float,
    beta2: float,
    adam_eps: float,
    optimizer_mode: str,
    segment_tape_vjp_mode: str,
    tape_mode: str,
    site_initialization: str = SITE_INITIALIZATION_LEGACY_SPARSE,
    edit_block_size: int = 4,
    allow_repeat_loaded_frames: bool = False,
    endpoint_record_source: str = "slow-owner-run",
    gate4_time_slabs: int = 1,
    gate4_residual_depth_padding: float = 0.001,
    experimental_native_cut_prep_delta: bool = False,
    experimental_native_sorted_delta: bool = False,
    experimental_minimal_packed_delta_device: bool = False,
    experimental_cpu_rebase_delta: bool = False,
    experimental_kernel_order_packed_delta_device: bool = False,
    experimental_smallrun16_packed_delta: bool = False,
    experimental_launch_only_packed_delta: bool = False,
    experimental_unchecked_launch_only_packed_delta: bool = False,
    experimental_reduce32_launch_only_packed_delta: bool = False,
    experimental_rowselect32_launch_only_packed_delta: bool = False,
    experimental_rowdesc_launch_only_packed_delta: bool = False,
    experimental_rowdesc32_launch_only_packed_delta: bool = False,
    experimental_native_pack_records: bool = False,
    experimental_native_emitted_pack_records: bool = False,
    experimental_selected_only_owner_run_delta_prep: bool = False,
    experimental_native_owner_run_cutwalk_delta: bool = False,
    defer_heldout_device: bool = False,
    partial_out_json: Path | None = None,
    post_run_benchmark_environment_settle_s: float = 0.0,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if experimental_unchecked_launch_only_packed_delta and not experimental_launch_only_packed_delta:
        raise ValueError(
            "--experimental-unchecked-launch-only-packed-delta requires --experimental-launch-only-packed-delta"
        )
    if experimental_rowdesc_launch_only_packed_delta and not experimental_launch_only_packed_delta:
        raise ValueError(
            "--experimental-rowdesc-launch-only-packed-delta requires --experimental-launch-only-packed-delta"
        )
    if experimental_reduce32_launch_only_packed_delta and not experimental_launch_only_packed_delta:
        raise ValueError(
            "--experimental-reduce32-launch-only-packed-delta requires --experimental-launch-only-packed-delta"
        )
    if experimental_rowselect32_launch_only_packed_delta and not experimental_launch_only_packed_delta:
        raise ValueError(
            "--experimental-rowselect32-launch-only-packed-delta requires --experimental-launch-only-packed-delta"
        )
    if experimental_rowselect32_launch_only_packed_delta and experimental_reduce32_launch_only_packed_delta:
        raise ValueError(
            "--experimental-rowselect32-launch-only-packed-delta cannot be combined with "
            "--experimental-reduce32-launch-only-packed-delta"
        )
    if experimental_rowselect32_launch_only_packed_delta and experimental_rowdesc_launch_only_packed_delta:
        raise ValueError(
            "--experimental-rowselect32-launch-only-packed-delta cannot be combined with "
            "--experimental-rowdesc-launch-only-packed-delta"
        )
    if experimental_reduce32_launch_only_packed_delta and experimental_rowdesc_launch_only_packed_delta:
        raise ValueError(
            "--experimental-reduce32-launch-only-packed-delta cannot be combined with "
            "--experimental-rowdesc-launch-only-packed-delta"
        )
    if experimental_rowdesc32_launch_only_packed_delta and not experimental_rowdesc_launch_only_packed_delta:
        raise ValueError(
            "--experimental-rowdesc32-launch-only-packed-delta requires "
            "--experimental-rowdesc-launch-only-packed-delta"
        )
    benchmark_environment_start = _capture_benchmark_environment()
    rows = []
    for frame_count in frame_counts:
        print(
            f"[train_eval_owner_run_tape] start tape_mode={tape_mode} frame_count={frame_count}",
            flush=True,
        )
        row = _run_one(
            config_path=config_path,
            frame_count=frame_count,
            render_size=render_size,
            site_count=site_count,
            near=near,
            far=far,
            density=density,
            site_initialization=site_initialization,
            invalid_epsilon=invalid_epsilon,
            transmittance_threshold=transmittance_threshold,
            synthetic_motion=synthetic_motion,
            steps=steps,
            warmup_steps=warmup_steps,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            adam_eps=adam_eps,
            optimizer_mode=optimizer_mode,
            segment_tape_vjp_mode=segment_tape_vjp_mode,
            tape_mode=tape_mode,
            edit_block_size=edit_block_size,
            allow_repeat_loaded_frames=allow_repeat_loaded_frames,
            endpoint_record_source=endpoint_record_source,
            gate4_time_slabs=gate4_time_slabs,
            gate4_residual_depth_padding=gate4_residual_depth_padding,
            experimental_native_cut_prep_delta=experimental_native_cut_prep_delta,
            experimental_native_sorted_delta=experimental_native_sorted_delta,
            experimental_minimal_packed_delta_device=experimental_minimal_packed_delta_device,
            experimental_cpu_rebase_delta=experimental_cpu_rebase_delta,
            experimental_kernel_order_packed_delta_device=experimental_kernel_order_packed_delta_device,
            experimental_smallrun16_packed_delta=experimental_smallrun16_packed_delta,
            experimental_launch_only_packed_delta=experimental_launch_only_packed_delta,
            experimental_unchecked_launch_only_packed_delta=experimental_unchecked_launch_only_packed_delta,
            experimental_reduce32_launch_only_packed_delta=experimental_reduce32_launch_only_packed_delta,
            experimental_rowselect32_launch_only_packed_delta=experimental_rowselect32_launch_only_packed_delta,
            experimental_rowdesc_launch_only_packed_delta=experimental_rowdesc_launch_only_packed_delta,
            experimental_rowdesc32_launch_only_packed_delta=experimental_rowdesc32_launch_only_packed_delta,
            experimental_native_pack_records=experimental_native_pack_records,
            experimental_native_emitted_pack_records=experimental_native_emitted_pack_records,
            experimental_selected_only_owner_run_delta_prep=experimental_selected_only_owner_run_delta_prep,
            experimental_native_owner_run_cutwalk_delta=experimental_native_owner_run_cutwalk_delta,
            defer_heldout_device=defer_heldout_device,
        )
        rows.append(row)
        print(
            "[train_eval_owner_run_tape] done "
            f"frame_count={frame_count} status={row['status']} "
            f"total_ms={row['step_summary']['total']['mean_s'] * 1000.0:.3f} "
            f"render_ms={row['step_summary']['render']['mean_s'] * 1000.0:.3f} "
            f"backward_ms={row['step_summary']['backward']['mean_s'] * 1000.0:.3f}",
            flush=True,
        )
        if partial_out_json is not None:
            partial_out_json.parent.mkdir(parents=True, exist_ok=True)
            partial_out_json.write_text(
                json.dumps(
                    {
                        "benchmark": "world_foam_lane2_segment_tape_train_eval_mps_partial",
                        "status": "running",
                        "frame_counts_requested": list(frame_counts),
                        "frame_counts_completed": [int(row["frame_count"]) for row in rows],
                        "render_size": render_size,
                        "site_count": site_count,
                        "tape_mode": tape_mode,
                        "endpoint_record_source": (
                            "gate4-affine"
                            if _is_gate4_affine_candidate_fused_mse_mode(tape_mode)
                            else endpoint_record_source
                        ),
                        "gate4_time_slabs": int(gate4_time_slabs),
                        "gate4_residual_depth_padding": float(gate4_residual_depth_padding),
                        "gate4_affine_candidate_csr_fused_mse": bool(
                            _is_gate4_affine_candidate_fused_mse_mode(tape_mode)
                        ),
                        "gate4_affine_candidate_csr_trackmse_fused_mse": bool(
                            _is_gate4_affine_candidate_trackmse_mode(tape_mode)
                        ),
                        "gate4_affine_candidate_csr_coeff16_fused_mse": bool(
                            _is_gate4_affine_candidate_coeff16_mode(tape_mode)
                        ),
                        "gate4_affine_candidate_csr_cap224_fused_mse": bool(
                            _is_gate4_affine_candidate_cap224_mode(tape_mode)
                        ),
                        "gate4_affine_candidate_csr_densitymask_fused_mse": bool(
                            _is_gate4_affine_candidate_densitymask_mode(tape_mode)
                        ),
                        "gate4_affine_candidate_csr_sample_reduce_fused_mse": bool(
                            _is_gate4_affine_candidate_sample_reduce_mode(tape_mode)
                        ),
                        "gate4_affine_candidate_csr_sortnet_fused_mse": bool(
                            _is_gate4_affine_candidate_sortnet_mode(tape_mode)
                        ),
                        "gate4_affine_candidate_csr_framegroup16_cached_fused_mse": bool(
                            _is_gate4_affine_candidate_framegroup16_cached_mode(tape_mode)
                        ),
                        "gate4_affine_candidate_csr_sitecache_fused_mse": bool(
                            _is_gate4_affine_candidate_sitecache_mode(tape_mode)
                        ),
                        "gate4_affine_candidate_csr_ownerupdate_fused_mse": bool(
                            _is_gate4_affine_candidate_ownerupdate_mode(tape_mode)
                        ),
                        "gate4_affine_candidate_csr_ownerupdate_i16_fused_mse": bool(
                            _is_gate4_affine_candidate_ownerupdate_i16_mode(tape_mode)
                        ),
                        "gate4_affine_candidate_csr_ownerkeep_fused_mse": bool(
                            _is_gate4_affine_candidate_ownerkeep_mode(tape_mode)
                        ),
                        "gate4_affine_candidate_csr_ownerkeep_i16_fused_mse": bool(
                            _is_gate4_affine_candidate_ownerkeep_i16_mode(tape_mode)
                        ),
                        "experimental_native_cut_prep_delta": bool(experimental_native_cut_prep_delta),
                        "experimental_native_sorted_delta": bool(experimental_native_sorted_delta),
                        "experimental_minimal_packed_delta_device": bool(experimental_minimal_packed_delta_device),
                        "experimental_smallrun16_packed_delta": bool(experimental_smallrun16_packed_delta),
                        "experimental_launch_only_packed_delta": bool(experimental_launch_only_packed_delta),
                        "experimental_unchecked_launch_only_packed_delta": bool(
                            experimental_unchecked_launch_only_packed_delta
                        ),
                        "experimental_reduce32_launch_only_packed_delta": bool(
                            experimental_reduce32_launch_only_packed_delta
                        ),
                        "experimental_rowdesc_launch_only_packed_delta": bool(
                            experimental_rowdesc_launch_only_packed_delta
                        ),
                        "experimental_rowdesc32_launch_only_packed_delta": bool(
                            experimental_rowdesc32_launch_only_packed_delta
                        ),
                        "experimental_native_pack_records": bool(experimental_native_pack_records),
                        "experimental_native_emitted_pack_records": bool(experimental_native_emitted_pack_records),
                        "experimental_selected_only_owner_run_delta_prep": bool(
                            experimental_selected_only_owner_run_delta_prep
                        ),
                        "experimental_native_owner_run_cutwalk_delta": bool(
                            experimental_native_owner_run_cutwalk_delta
                        ),
                        "defer_heldout_device": bool(defer_heldout_device),
                        "edit_block_size": int(edit_block_size),
                        "optimizer_mode": optimizer_mode,
                        "allow_repeat_loaded_frames": bool(allow_repeat_loaded_frames),
                        "benchmark_environment": {
                            "status": benchmark_environment_start.get("status", "unchecked"),
                            "start": benchmark_environment_start,
                        },
                        "rows": rows,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    frame_scale = float(frame_counts[-1]) / float(max(frame_counts[0], 1))
    total_scale = _ratio_last_first(rows[-1]["step_summary"]["total"]["mean_s"], rows[0]["step_summary"]["total"]["mean_s"])
    render_scale = _ratio_last_first(
        rows[-1]["step_summary"]["render"]["mean_s"],
        rows[0]["step_summary"]["render"]["mean_s"],
    )
    backward_scale = _ratio_last_first(
        rows[-1]["step_summary"]["backward"]["mean_s"],
        rows[0]["step_summary"]["backward"]["mean_s"],
    )
    has_fused_loss_vjp_timing = all("fused_loss_vjp" in row["step_summary"] for row in rows)
    fused_loss_vjp_scale = (
        _ratio_last_first(
            rows[-1]["step_summary"]["fused_loss_vjp"]["mean_s"],
            rows[0]["step_summary"]["fused_loss_vjp"]["mean_s"],
        )
        if has_fused_loss_vjp_timing
        else None
    )
    render_timing_scope = (
        "fused_loss_vjp_includes_render"
        if has_fused_loss_vjp_timing
        and all(abs(float(row["step_summary"]["render"]["mean_s"])) <= 1.0e-12 for row in rows)
        else "separate_render"
    )
    selected_tape_segment_scale = rows[-1]["train_selected_tape_segments"] / max(
        rows[0]["train_selected_tape_segments"], 1
    )
    selected_tape_storage_scale = rows[-1]["train_selected_tape_storage_bytes"] / max(
        rows[0]["train_selected_tape_storage_bytes"], 1
    )
    selected_tape_mps_resident_storage_scale = rows[-1]["train_selected_tape_mps_resident_storage_bytes"] / max(
        rows[0]["train_selected_tape_mps_resident_storage_bytes"], 1
    )
    selected_tape_mps_resident_noncoeff_storage_scale = rows[-1][
        "train_selected_tape_mps_resident_noncoeff_storage_bytes"
    ] / max(rows[0]["train_selected_tape_mps_resident_noncoeff_storage_bytes"], 1)
    endpoint_record_coeff_mps_resident_storage_scale = rows[-1][
        "train_endpoint_record_coeff_mps_resident_storage_bytes"
    ] / max(rows[0]["train_endpoint_record_coeff_mps_resident_storage_bytes"], 1)
    endpoint_record_edit_op_scale = rows[-1]["train_endpoint_record_edit_ops"] / max(
        rows[0]["train_endpoint_record_edit_ops"], 1
    )
    has_scale = len(rows) > 1
    acceptance = {
        "all_rows_ok": all(row["status"] == "ok" for row in rows),
        "total_step_sublinear_vs_frames": (not has_scale) or total_scale < frame_scale,
        "render_sublinear_vs_frames": (not has_scale) or render_scale < frame_scale,
        "backward_sublinear_vs_frames": (not has_scale) or backward_scale < frame_scale,
        "selected_tape_segments_below_full_at_max_frame": rows[-1]["train_selected_tape_segments_vs_full"] <= 1.0,
        "selected_tape_storage_below_full_at_max_frame": rows[-1]["train_selected_tape_storage_vs_full"] <= 1.0,
        "owner_run_segments_below_full_at_max_frame": rows[-1]["train_selected_tape_segments_vs_full"] <= 1.0,
    }
    if fused_loss_vjp_scale is not None:
        acceptance["fused_loss_vjp_sublinear_vs_frames"] = (not has_scale) or fused_loss_vjp_scale < frame_scale
    objective_adapters = [
        row["world_foam_objective_adapter"]
        for row in rows
        if isinstance(row.get("world_foam_objective_adapter"), dict)
    ]
    world_foam_objective_adapter = objective_adapters[0] if objective_adapters else None
    benchmark_environment = _merge_benchmark_environments_with_optional_settle(
        benchmark_environment_start,
        settle_s=float(post_run_benchmark_environment_settle_s),
    )
    return {
        "benchmark": "world_foam_lane2_segment_tape_train_eval_mps",
        "status": "ok" if all(acceptance.values()) else "failed",
        "benchmark_environment": benchmark_environment,
        "gate": f"{tape_mode}_compact_segment_tape_rgb_only_site_rgba_train_eval",
        "device": "mps",
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": render_size,
        "site_count": site_count,
        "site_initialization": site_initialization,
        "tape_mode": tape_mode,
        "endpoint_record_source": (
            "gate4-affine"
            if _is_gate4_affine_candidate_fused_mse_mode(tape_mode)
            else endpoint_record_source
        ),
        "gate4_time_slabs": int(gate4_time_slabs),
        "gate4_residual_depth_padding": float(gate4_residual_depth_padding),
        "gate4_affine_candidate_csr_fused_mse": bool(
            _is_gate4_affine_candidate_fused_mse_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_trackmse_fused_mse": bool(
            _is_gate4_affine_candidate_trackmse_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_coeff16_fused_mse": bool(
            _is_gate4_affine_candidate_coeff16_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_cap224_fused_mse": bool(
            _is_gate4_affine_candidate_cap224_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_densitymask_fused_mse": bool(
            _is_gate4_affine_candidate_densitymask_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_sample_reduce_fused_mse": bool(
            _is_gate4_affine_candidate_sample_reduce_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_sortnet_fused_mse": bool(_is_gate4_affine_candidate_sortnet_mode(tape_mode)),
        "gate4_affine_candidate_csr_framegroup16_cached_fused_mse": bool(
            _is_gate4_affine_candidate_framegroup16_cached_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_sitecache_fused_mse": bool(
            _is_gate4_affine_candidate_sitecache_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_ownerupdate_fused_mse": bool(
            _is_gate4_affine_candidate_ownerupdate_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_ownerupdate_i16_fused_mse": bool(
            _is_gate4_affine_candidate_ownerupdate_i16_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_ownerkeep_fused_mse": bool(
            _is_gate4_affine_candidate_ownerkeep_mode(tape_mode)
        ),
        "gate4_affine_candidate_csr_ownerkeep_i16_fused_mse": bool(
            _is_gate4_affine_candidate_ownerkeep_i16_mode(tape_mode)
        ),
        "experimental_native_cut_prep_delta": bool(experimental_native_cut_prep_delta),
        "experimental_native_sorted_delta": bool(experimental_native_sorted_delta),
        "experimental_cpu_rebase_delta": bool(experimental_cpu_rebase_delta),
        "experimental_minimal_packed_delta_device": bool(experimental_minimal_packed_delta_device),
        "experimental_kernel_order_packed_delta_device": bool(experimental_kernel_order_packed_delta_device),
        "experimental_smallrun16_packed_delta": bool(experimental_smallrun16_packed_delta),
        "experimental_launch_only_packed_delta": bool(experimental_launch_only_packed_delta),
        "experimental_unchecked_launch_only_packed_delta": bool(experimental_unchecked_launch_only_packed_delta),
        "experimental_reduce32_launch_only_packed_delta": bool(experimental_reduce32_launch_only_packed_delta),
        "experimental_rowselect32_launch_only_packed_delta": bool(experimental_rowselect32_launch_only_packed_delta),
        "experimental_rowdesc_launch_only_packed_delta": bool(experimental_rowdesc_launch_only_packed_delta),
        "experimental_rowdesc32_launch_only_packed_delta": bool(experimental_rowdesc32_launch_only_packed_delta),
        "experimental_native_pack_records": bool(experimental_native_pack_records),
        "experimental_native_emitted_pack_records": bool(experimental_native_emitted_pack_records),
        "experimental_selected_only_owner_run_delta_prep": bool(experimental_selected_only_owner_run_delta_prep),
        "experimental_native_owner_run_cutwalk_delta": bool(experimental_native_owner_run_cutwalk_delta),
        "defer_heldout_device": bool(defer_heldout_device),
        "timed_mps_residency_scope": (
            "train_tape_targets_site_only" if defer_heldout_device else "train_and_heldout_tapes_targets_site"
        ),
        "tape_modes_resolved": [str(row.get("tape_mode_resolved", tape_mode)) for row in rows],
        "auto_selector_policy": (
            {
                "packed_mode": DELTA_PACKED_FRAMEGROUP16_MODE,
                "smallrun16_mode": DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
                "i16x3_mode": DELTA_I16X3_FRAMEGROUP16_MODE,
                "packed_max_frame_count": DELTA_AUTO_PACKED_MAX_FRAME_COUNT,
                "prefer_smallrun16": bool(experimental_smallrun16_packed_delta),
                "prefer_launch_only_packed": bool(experimental_launch_only_packed_delta),
                "prefer_unchecked_launch_only_packed": bool(experimental_unchecked_launch_only_packed_delta),
                "prefer_reduce32_launch_only_packed": bool(experimental_reduce32_launch_only_packed_delta),
                "prefer_rowselect32_launch_only_packed": bool(experimental_rowselect32_launch_only_packed_delta),
                "prefer_rowdesc_launch_only_packed": bool(experimental_rowdesc_launch_only_packed_delta),
                "prefer_rowdesc32_launch_only_packed": bool(experimental_rowdesc32_launch_only_packed_delta),
            }
            if tape_mode == DELTA_AUTO_FRAMEGROUP16_MODE
            else None
        ),
        "edit_block_size": int(edit_block_size),
        "allow_repeat_loaded_frames": bool(allow_repeat_loaded_frames),
        "repeat_loaded_frames": any(bool(row.get("repeat_loaded_frames")) for row in rows),
        "synthetic_motion": synthetic_motion.to_dict(),
        "gradient_scope": f"frozen_geometry_{tape_mode}_rgb_only_site_rgba_{optimizer_mode}",
        "world_foam_objective_adapter": world_foam_objective_adapter,
        "world_foam_objective_adapter_rows_all_match": (
            bool(world_foam_objective_adapter)
            and len(objective_adapters) == len(rows)
            and all(adapter == world_foam_objective_adapter for adapter in objective_adapters)
        ),
        "optimizer_mode": optimizer_mode,
        "segment_tape_vjp_mode": segment_tape_vjp_mode if optimizer_mode == "autograd" else "manual_direct_atomic_grad_only",
        "full_trainer_claim": False,
        "full_geometry_gradient_claim": False,
        "quality_claim": False,
        "density_independent_depth_claim": tape_mode in ENDPOINT_SEMANTIC_TAPE_MODES,
        "continuous_absorption_depth_semantic": tape_mode in ENDPOINT_SEMANTIC_TAPE_MODES,
        "transmittance_threshold": transmittance_threshold,
        "acceptance": acceptance,
        "render_timing_scope": render_timing_scope,
        "frame_scale_first_to_last": frame_scale,
        "total_step_scale_first_to_last": total_scale,
        "render_scale_first_to_last": render_scale,
        "backward_scale_first_to_last": backward_scale,
        "fused_loss_vjp_scale_first_to_last": fused_loss_vjp_scale,
        "selected_tape_segment_scale_first_to_last": selected_tape_segment_scale,
        "selected_tape_storage_scale_first_to_last": selected_tape_storage_scale,
        "selected_tape_mps_resident_storage_scale_first_to_last": selected_tape_mps_resident_storage_scale,
        "selected_tape_mps_resident_noncoeff_storage_scale_first_to_last": (
            selected_tape_mps_resident_noncoeff_storage_scale
        ),
        "endpoint_record_coeff_mps_resident_storage_scale_first_to_last": (
            endpoint_record_coeff_mps_resident_storage_scale
        ),
        "endpoint_record_edit_op_scale_first_to_last": endpoint_record_edit_op_scale,
        "owner_run_segment_scale_first_to_last": selected_tape_segment_scale,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/eval same-owner run segment-tape World Foam RGB path on MPS.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--density", type=float, default=10.0)
    parser.add_argument(
        "--site-initialization",
        choices=SITE_INITIALIZATION_CHOICES,
        default=SITE_INITIALIZATION_LEGACY_SPARSE,
        help="Deterministic train-ray site seeding strategy. Default preserves legacy benchmark artifacts.",
    )
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--optimizer-mode", choices=("manual-vjp", "autograd"), default="autograd")
    parser.add_argument("--edit-block-size", type=int, default=4)
    parser.add_argument(
        "--tape-mode",
        choices=(
            "owner-run",
            OWNER_RUN_FUSED_MSE_MODE,
            OWNER_RUN_FUSED_MSE_NOMID_MODE,
            "active-internal",
            "full",
            "endpoint-run",
            ENDPOINT_RUN_FUSED_MSE_MODE,
            "endpoint-record-edit",
            "endpoint-record-edit-fused-mse",
            "endpoint-record-edit-coeff16-fused-mse",
            "endpoint-record-delta-replace-coeff16-fused-mse",
            "endpoint-record-delta-replace-coeff16-i16x3-fused-mse",
            DELTA_I16X3_FRAMEGROUP16_MODE,
            DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
            "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
            "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse",
            "endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse",
            DELTA_PACKED_SCALAR_MODE,
            DELTA_PACKED_FRAMEGROUP16_MODE,
            DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
            DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
            DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
            OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            DELTA_AUTO_FRAMEGROUP16_MODE,
            "endpoint-record-delta-replace-coeff16-i16x4-fused-mse",
            "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
            "endpoint-record-edit-block4",
            "endpoint-record-edit-block-coeff",
            "endpoint-record-edit-block-coeff-rgb",
            "endpoint-record-edit-block-coeff-fused-mse",
            "endpoint-record-edit-block-coeff16",
            "endpoint-record-edit-block-coeff16-fused-mse",
            "endpoint-record-edit-block-coeff16-packed-fused-mse",
            "endpoint-record-edit-block-coeff16-i16-fused-mse",
            "endpoint-record-edit-block-coeff16-i16x3-fused-mse",
            GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_NUM32_DEN16_TRACK_MSE_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_COEFF16_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_COEFF16_CAP224_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_COEFF16_DENSITYMASK_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_COEFF16_SAMPLE_REDUCE_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_COEFF16_SORTNET_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_COEFF16_FRAMEGROUP16_CACHED_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_COEFF16_SITECACHE_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_I16_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_I16_FUSED_MSE_MODE,
            GATE4_AFFINE_CANDIDATE_COEFF16_TRACK_MSE_FUSED_MSE_MODE,
        ),
        default="owner-run",
    )
    parser.add_argument(
        "--segment-tape-vjp-mode",
        choices=("direct_atomic_grad_only", "direct_atomic_track"),
        default="direct_atomic_grad_only",
    )
    parser.add_argument(
        "--endpoint-record-source",
        choices=("slow-owner-run", "gate4-affine"),
        default="slow-owner-run",
        help="Source used to build endpoint-record rows for endpoint-record tape modes.",
    )
    parser.add_argument("--gate4-time-slabs", type=int, default=1)
    parser.add_argument("--gate4-residual-depth-padding", type=float, default=0.001)
    parser.add_argument(
        "--experimental-native-cut-prep-delta",
        action="store_true",
        help=(
            "Use the experimental native sorted-to-cut-array prep, then the promoted native cut-array "
            "delta builder. This removes Python cut-row assembly without using the slow sorted final packer."
        ),
    )
    parser.add_argument(
        "--experimental-native-sorted-delta",
        action="store_true",
        help=(
            "Use the experimental exact native sorted-row Gate4 delta builder. "
            "Default remains the native cut-array keeper until full train/eval timing promotes this path."
        ),
    )
    parser.add_argument(
        "--experimental-minimal-packed-delta-device",
        action="store_true",
        help=(
            "For packed framegroup16 fused-MSE, keep only the tensors read by the warm kernel on MPS; "
            "build the full endpoint-record replay device lazily for final PSNR rendering."
        ),
    )
    parser.add_argument(
        "--experimental-kernel-order-packed-delta-device",
        action="store_true",
        help=(
            "For packed framegroup16 fused-MSE, allocate only warm-kernel tensors on MPS and create them "
            "in kernel buffer order. This is a diagnostic for MPS allocation-order effects."
        ),
    )
    parser.add_argument(
        "--experimental-smallrun16-packed-delta",
        action="store_true",
        help=(
            "When auto framegroup16 resolves to the packed path, use the smallrun16 packed shader. "
            "The run fails fast if any selected row needs more than 16 segments."
        ),
    )
    parser.add_argument(
        "--experimental-launch-only-packed-delta",
        action="store_true",
        help=(
            "For the default packed framegroup16 fused-MSE path, use the native launch-only op that trusts "
            "prepare-time tape validation and skips per-launch MPS-to-CPU config/offset checks."
        ),
    )
    parser.add_argument(
        "--experimental-unchecked-launch-only-packed-delta",
        action="store_true",
        help=(
            "With --experimental-launch-only-packed-delta, use the default packed framegroup16 op variant "
            "that skips native per-launch dtype/shape checks after prepare-time validation."
        ),
    )
    parser.add_argument(
        "--experimental-reduce32-launch-only-packed-delta",
        action="store_true",
        help=(
            "With --experimental-launch-only-packed-delta, use the compact chunk-offset packed "
            "framegroup16 shader fork with 32 site-gradient reduction slots."
        ),
    )
    parser.add_argument(
        "--experimental-rowselect32-launch-only-packed-delta",
        action="store_true",
        help=(
            "With --experimental-launch-only-packed-delta, use the compact chunk-offset packed "
            "framegroup16 shader fork that computes each frame lane row selection locally."
        ),
    )
    parser.add_argument(
        "--experimental-rowdesc-launch-only-packed-delta",
        action="store_true",
        help=(
            "With --experimental-launch-only-packed-delta, use the compact per-track-frame row descriptor "
            "packed framegroup16 op variant instead of chunk change offsets."
        ),
    )
    parser.add_argument(
        "--experimental-rowdesc32-launch-only-packed-delta",
        action="store_true",
        help=(
            "With --experimental-rowdesc-launch-only-packed-delta, use the row descriptor shader fork "
            "with 32 threadgroup site-gradient reduction slots for site counts up to 32."
        ),
    )
    parser.add_argument(
        "--experimental-cpu-rebase-delta",
        action="store_true",
        help=(
            "Clone endpoint delta tensors into fresh contiguous CPU tensors before MPS transfer. "
            "This isolates tensor provenance/allocation effects without changing endpoint math."
        ),
    )
    parser.add_argument(
        "--experimental-native-pack-records",
        action="store_true",
        help=(
            "Use the native C++ CPU endpoint-record packer for packed delta fused-MSE modes. "
            "Default keeps the Python/Torch packing path until this full train/eval gate promotes it."
        ),
    )
    parser.add_argument(
        "--experimental-native-emitted-pack-records",
        action="store_true",
        help=(
            "Ask the native Gate4 cut-array delta row walk to emit packed endpoint records directly, "
            "then feed those records to packed delta fused-MSE modes without a separate pack pass."
        ),
    )
    parser.add_argument(
        "--experimental-selected-only-owner-run-delta-prep",
        action="store_true",
        help=(
            "For slow-owner-run owner-run delta packed modes, skip full/active/endpoint baseline "
            "segment-tape construction and prepare only the selected owner-run delta tape. Artifacts "
            "from this mode are timing/correctness evidence, not full-vs-selected storage proof."
        ),
    )
    parser.add_argument(
        "--experimental-native-owner-run-cutwalk-delta",
        action="store_true",
        help=(
            "For slow-owner-run owner-run delta packed modes, build cut arrays in Python but use the "
            "native cut-array owner-transition walker to emit the delta tape instead of materializing "
            "Python owner-run record sequences. Experimental: records are full cutwalk rows and must be "
            "checked against shader output parity before promotion."
        ),
    )
    parser.add_argument(
        "--defer-heldout-device",
        action="store_true",
        help=(
            "Prepare heldout MPS tape/targets only after the timed optimizer loop. "
            "This keeps shader timing scoped to train tape, train targets, and site parameters."
        ),
    )
    parser.add_argument(
        "--repeat-loaded-frames",
        action="store_true",
        help=(
            "Repeat a shorter loaded view-major fixture when requested frame counts exceed the real fixture. "
            "This is a synthetic speed-scaling smoke, not a real longer-video quality run."
        ),
    )
    parser.add_argument(
        "--benchmark-environment-check-only",
        action="store_true",
        help=(
            "Print the benchmark environment process snapshot and exit. "
            "Exit status is nonzero when blocking processes would contaminate timing promotion."
        ),
    )
    parser.add_argument(
        "--require-benchmark-environment-ok",
        action="store_true",
        help=(
            "Fail before building tapes when benchmark_environment.status is contended. "
            "Use this for promotion runs so high-CPU Python or Torch/Metal/MPS/PyTest competitors do not waste a sweep."
        ),
    )
    parser.add_argument(
        "--wait-for-benchmark-environment-ok-timeout-s",
        type=float,
        default=0.0,
        help=(
            "When combined with --require-benchmark-environment-ok or --benchmark-environment-check-only, "
            "poll until the environment is promotable or this timeout expires."
        ),
    )
    parser.add_argument(
        "--wait-for-benchmark-environment-ok-poll-s",
        type=float,
        default=15.0,
        help="Poll interval for --wait-for-benchmark-environment-ok-timeout-s.",
    )
    parser.add_argument(
        "--post-run-benchmark-environment-settle-s",
        type=float,
        default=0.0,
        help=(
            "When the immediate post-run blocker is only MTLCompilerService, wait this long and "
            "use a second environment snapshot for promotion. Other Python/Torch/MPS blockers "
            "still make the artifact diagnostic."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_owner_run_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json",
    )
    parser.add_argument("--partial-out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.benchmark_environment_check_only:
        environment = _wait_for_benchmark_environment_ok(
            timeout_s=float(args.wait_for_benchmark_environment_ok_timeout_s),
            poll_s=float(args.wait_for_benchmark_environment_ok_poll_s),
        )
        _print_benchmark_environment(environment)
        if _benchmark_environment_blocks_promotion(environment):
            raise SystemExit(2)
        return
    if args.require_benchmark_environment_ok:
        environment = _wait_for_benchmark_environment_ok(
            timeout_s=float(args.wait_for_benchmark_environment_ok_timeout_s),
            poll_s=float(args.wait_for_benchmark_environment_ok_poll_s),
        )
        if _benchmark_environment_blocks_promotion(environment):
            _print_benchmark_environment(environment)
            raise SystemExit(2)
    payload = run_train_eval(
        config_path=args.config,
        frame_counts=_parse_int_list(args.frame_counts),
        render_size=args.render_size,
        site_count=args.site_count,
        near=args.near,
        far=args.far,
        density=args.density,
        site_initialization=args.site_initialization,
        invalid_epsilon=args.invalid_epsilon,
        transmittance_threshold=args.transmittance_threshold,
        synthetic_motion=SyntheticRayMotion(
            origin_velocity=(args.origin_velocity_x, args.origin_velocity_y, args.origin_velocity_z),
            direction_velocity=(args.direction_velocity_x, args.direction_velocity_y, args.direction_velocity_z),
        ),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        optimizer_mode=args.optimizer_mode,
        segment_tape_vjp_mode=args.segment_tape_vjp_mode,
        tape_mode=args.tape_mode,
        edit_block_size=int(args.edit_block_size),
        allow_repeat_loaded_frames=bool(args.repeat_loaded_frames),
        endpoint_record_source=args.endpoint_record_source,
        gate4_time_slabs=int(args.gate4_time_slabs),
        gate4_residual_depth_padding=float(args.gate4_residual_depth_padding),
        experimental_native_cut_prep_delta=bool(args.experimental_native_cut_prep_delta),
        experimental_native_sorted_delta=bool(args.experimental_native_sorted_delta),
        experimental_minimal_packed_delta_device=bool(args.experimental_minimal_packed_delta_device),
        experimental_cpu_rebase_delta=bool(args.experimental_cpu_rebase_delta),
        experimental_kernel_order_packed_delta_device=bool(args.experimental_kernel_order_packed_delta_device),
        experimental_smallrun16_packed_delta=bool(args.experimental_smallrun16_packed_delta),
        experimental_launch_only_packed_delta=bool(args.experimental_launch_only_packed_delta),
        experimental_unchecked_launch_only_packed_delta=bool(args.experimental_unchecked_launch_only_packed_delta),
        experimental_reduce32_launch_only_packed_delta=bool(args.experimental_reduce32_launch_only_packed_delta),
        experimental_rowselect32_launch_only_packed_delta=bool(args.experimental_rowselect32_launch_only_packed_delta),
        experimental_rowdesc_launch_only_packed_delta=bool(args.experimental_rowdesc_launch_only_packed_delta),
        experimental_rowdesc32_launch_only_packed_delta=bool(args.experimental_rowdesc32_launch_only_packed_delta),
        experimental_native_pack_records=bool(args.experimental_native_pack_records),
        experimental_native_emitted_pack_records=bool(args.experimental_native_emitted_pack_records),
        experimental_selected_only_owner_run_delta_prep=bool(
            args.experimental_selected_only_owner_run_delta_prep
        ),
        experimental_native_owner_run_cutwalk_delta=bool(args.experimental_native_owner_run_cutwalk_delta),
        defer_heldout_device=bool(args.defer_heldout_device),
        partial_out_json=args.partial_out_json,
        post_run_benchmark_environment_settle_s=float(args.post_run_benchmark_environment_settle_s),
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)
    if args.require_benchmark_environment_ok and _benchmark_environment_blocks_promotion(
        payload.get("benchmark_environment", {})
    ):
        print(
            "benchmark environment became contended during run; artifact is diagnostic, not promotable",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
