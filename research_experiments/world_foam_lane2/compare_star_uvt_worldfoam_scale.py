#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DYNAWORLD = Path(__file__).resolve().parents[2]
STAR_BENCHMARKS = (
    DYNAWORLD
    / "third_party"
    / "fast-mac-gsplat"
    / "variants"
    / "star_uvt_v0"
    / "research_project"
    / "benchmarks"
)
RESULTS_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2" / "results"
DEFAULT_VIDEO = DYNAWORLD / "data" / "youtube_curated_spans" / "high_motion_smokes" / "hlaZbH_OFBU_seg_003_4fps_16f.mp4"
DEFAULT_WORLDFOAM_ARTIFACT = (
    RESULTS_DIR
    / "2026-05-18_gate4_affine_tape_train_eval_fusedmse_affineclear_repeat20_render32_site12_2_4_8_16.json"
)
GATE4_AFFINE_CANDIDATE_CSR_MODE = "gate4-affine-candidate-num32-den16-fused-mse"
GATE4_AFFINE_CANDIDATE_CSR_TRACKMSE_MODE = "gate4-affine-candidate-num32-den16-trackmse-fused-mse"
GATE4_AFFINE_CANDIDATE_CSR_COEFF16_MODE = "gate4-affine-candidate-coeff16-fused-mse"
GATE4_AFFINE_CANDIDATE_CSR_COEFF16_CAP224_MODE = "gate4-affine-candidate-coeff16-cap224-fused-mse"
GATE4_AFFINE_CANDIDATE_CSR_COEFF16_DENSITYMASK_MODE = "gate4-affine-candidate-coeff16-densitymask-fused-mse"
GATE4_AFFINE_CANDIDATE_CSR_COEFF16_SAMPLE_REDUCE_MODE = "gate4-affine-candidate-coeff16-samplereduce-fused-mse"
GATE4_AFFINE_CANDIDATE_CSR_COEFF16_SORTNET_MODE = "gate4-affine-candidate-coeff16-sortnet-fused-mse"
GATE4_AFFINE_CANDIDATE_CSR_COEFF16_FRAMEGROUP16_CACHED_MODE = (
    "gate4-affine-candidate-coeff16-framegroup16cached-fused-mse"
)
GATE4_AFFINE_CANDIDATE_CSR_COEFF16_SITECACHE_MODE = "gate4-affine-candidate-coeff16-sitecache-fused-mse"
GATE4_AFFINE_CANDIDATE_CSR_COEFF16_OWNERUPDATE_MODE = "gate4-affine-candidate-coeff16-ownerupdate-fused-mse"
GATE4_AFFINE_CANDIDATE_CSR_COEFF16_OWNERUPDATE_I16_MODE = (
    "gate4-affine-candidate-coeff16-ownerupdate-i16-fused-mse"
)
GATE4_AFFINE_CANDIDATE_CSR_COEFF16_OWNERKEEP_MODE = "gate4-affine-candidate-coeff16-ownerkeep-fused-mse"
GATE4_AFFINE_CANDIDATE_CSR_COEFF16_OWNERKEEP_I16_MODE = (
    "gate4-affine-candidate-coeff16-ownerkeep-i16-fused-mse"
)
GATE4_AFFINE_CANDIDATE_CSR_COEFF16_TRACKMSE_MODE = "gate4-affine-candidate-coeff16-trackmse-fused-mse"
OWNER_RUN_FACTORIZE_MODE = "owner-run-delta-packed-factorized-recompute-fused-mse-nomid"
OWNER_RUN_FRAMESELECT_MODE = "owner-run-delta-packed-factorized-frameselect-recompute-fused-mse-nomid"
OWNER_RUN_FRAMEBITMASK_MODE = "owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid"
GATE4_AFFINE_CANDIDATE_CSR_MODES = {
    GATE4_AFFINE_CANDIDATE_CSR_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_TRACKMSE_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_COEFF16_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_COEFF16_CAP224_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_COEFF16_DENSITYMASK_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_COEFF16_SAMPLE_REDUCE_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_COEFF16_SORTNET_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_COEFF16_FRAMEGROUP16_CACHED_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_COEFF16_SITECACHE_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_COEFF16_OWNERUPDATE_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_COEFF16_OWNERUPDATE_I16_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_COEFF16_OWNERKEEP_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_COEFF16_OWNERKEEP_I16_MODE,
    GATE4_AFFINE_CANDIDATE_CSR_COEFF16_TRACKMSE_MODE,
}
OWNER_RUN_FACTORIZE_MODES = {
    OWNER_RUN_FACTORIZE_MODE,
    OWNER_RUN_FRAMESELECT_MODE,
    OWNER_RUN_FRAMEBITMASK_MODE,
}


def parse_int_list(value: str) -> tuple[int, ...]:
    out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not out:
        raise ValueError("expected at least one frame count")
    return out


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _benchmark_keyword_matches(command: str, keyword: str) -> bool:
    if keyword == "python":
        return keyword in command
    return re.search(rf"(^|[^a-z0-9]){re.escape(keyword)}([^a-z0-9]|$)", command) is not None


def _benchmark_low_cpu_wrapper_is_background(command: str) -> bool:
    wrapper_markers = (
        "screen -dms",
        "login -pflq",
        "run_btc15m_overnight_shadow_monitor.py",
        "run_worldfoam_star_native_cutwalk_gate.py",
        "summarize_btc15m_overnight",
        "watch_schema.py --watch",
        "sky.server.server",
        "mtlcompilerservice",
    )
    return any(marker in command for marker in wrapper_markers)


def _benchmark_process_blocks_promotion(
    *,
    command: str,
    pcpu: float,
    blocking_cpu_threshold: float,
    hard_keywords: tuple[str, ...],
) -> bool:
    return _benchmark_process_block_reason(
        command=command,
        pcpu=pcpu,
        blocking_cpu_threshold=blocking_cpu_threshold,
        hard_keywords=hard_keywords,
    ) is not None


def _benchmark_process_block_reason(
    *,
    command: str,
    pcpu: float,
    blocking_cpu_threshold: float,
    hard_keywords: tuple[str, ...],
) -> str | None:
    if pcpu >= blocking_cpu_threshold:
        return "high_cpu"
    lowered = command.lower()
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


def capture_benchmark_environment() -> dict[str, Any]:
    keywords = ("python", "pytest", "torch", "metal", "mps", "modal")
    hard_keywords = ("pytest", "torch", "metal", "mps")
    blocking_cpu_threshold = 5.0
    own_pid = os.getpid()
    own_ppid = os.getppid()
    try:
        result = subprocess.run(
            ["ps", "-wwaxo", "pid=,ppid=,pcpu=,pmem=,command="],
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
            "blocking_processes": [],
            "background_processes": [],
            "contending_processes": [],
        }

    process_rows: list[tuple[int, int, float, float, str]] = []
    parent_by_pid: dict[int, int] = {}
    for line in result.stdout.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            pcpu = float(parts[2])
            pmem = float(parts[3])
        except ValueError:
            continue
        command = parts[4]
        process_rows.append((pid, ppid, pcpu, pmem, command))
        parent_by_pid[pid] = ppid

    ignored_pids = _benchmark_ignored_process_pids(
        own_pid=own_pid,
        own_ppid=own_ppid,
        parent_by_pid=parent_by_pid,
    )
    blocking_processes: list[dict[str, Any]] = []
    background_processes: list[dict[str, Any]] = []
    for pid, ppid, pcpu, pmem, command in process_rows:
        if pid in ignored_pids:
            continue
        lowered = command.lower()
        if not any(_benchmark_keyword_matches(lowered, keyword) for keyword in keywords):
            continue
        process = {
            "pid": pid,
            "ppid": ppid,
            "pcpu": pcpu,
            "pmem": pmem,
            "command": command[:240],
        }
        block_reason = _benchmark_process_block_reason(
            command=command,
            pcpu=pcpu,
            blocking_cpu_threshold=blocking_cpu_threshold,
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
        "blocking_processes": blocking_processes[:8],
        "background_processes": background_processes[:8],
        "contending_processes": blocking_processes[:8],
    }


def merge_benchmark_environments(start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    status = "contended" if "contended" in {start.get("status"), end.get("status")} else start.get("status", "unchecked")
    if status == "ok" and end.get("status") != "ok":
        status = str(end.get("status", "unchecked"))
    return {
        "status": status,
        "start": start,
        "end": end,
    }


def is_mtl_compiler_process(process: dict[str, Any]) -> bool:
    command = str(process.get("command", "")).lower()
    return "mtlcompilerservice" in command


def only_mtl_compiler_blocks(environment: dict[str, Any]) -> bool:
    blocking = environment.get("blocking_processes")
    if environment.get("status") != "contended" or not isinstance(blocking, list) or not blocking:
        return False
    return all(isinstance(process, dict) and is_mtl_compiler_process(process) for process in blocking)


def merge_benchmark_environments_with_optional_settle(
    start: dict[str, Any],
    *,
    settle_s: float,
) -> dict[str, Any]:
    immediate_end = capture_benchmark_environment()
    if settle_s <= 0.0 or not only_mtl_compiler_blocks(immediate_end):
        return merge_benchmark_environments(start, immediate_end)

    time.sleep(float(settle_s))
    settled_end = capture_benchmark_environment()
    merged = merge_benchmark_environments(start, settled_end)
    merged["end_immediate"] = immediate_end
    merged["post_run_settle_s"] = float(settle_s)
    merged["transient_mtl_compiler_settled"] = merged["status"] != "contended"
    return merged


def benchmark_environment_blocks_promotion(environment: dict[str, Any]) -> bool:
    return environment.get("status") not in {"ok", "background"}


def wait_for_benchmark_environment_ok(*, timeout_s: float, poll_s: float) -> dict[str, Any]:
    started_at = time.perf_counter()
    environment = capture_benchmark_environment()
    while benchmark_environment_blocks_promotion(environment):
        if timeout_s <= 0.0 or time.perf_counter() - started_at >= timeout_s:
            return environment
        print(
            "benchmark environment contended; waiting for a quiet window",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(max(0.1, poll_s))
        environment = capture_benchmark_environment()
    return environment


def scale(first: float | None, last: float | None) -> float | None:
    if first is None or last is None or first <= 0.0:
        return None
    return float(last) / float(first)


def summary_median_ms(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get("summary", {}).get(key, {}).get("median")
    return float(value) if isinstance(value, (float, int)) else None


def worldfoam_median_ms(row: dict[str, Any], phase: str) -> float | None:
    value = row.get("step_summary", {}).get(phase, {}).get("median_s")
    return float(value) * 1000.0 if isinstance(value, (float, int)) else None


def worldfoam_wall_s(row: dict[str, Any], key: str) -> float | None:
    value = row.get("wall_timing", {}).get(key)
    return float(value) if isinstance(value, (float, int)) else None


def _row_float(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (float, int)):
            return float(value)
    return 0.0


def _worldfoam_affine_ray_storage(row: dict[str, Any]) -> float:
    by_key = row.get("train_selected_tape_mps_resident_storage_by_key")
    if isinstance(by_key, dict) and isinstance(by_key.get("affine_ray_f32"), (float, int)):
        return float(by_key["affine_ray_f32"])
    return 0.0


def summarize_star_rows(star_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_frame = {int(row["frames"]): row for row in star_rows}
    frames = sorted(rows_by_frame)
    total_by_frame = {str(frame): summary_median_ms(rows_by_frame[frame], "total_ms") for frame in frames}
    backward_by_frame = {str(frame): summary_median_ms(rows_by_frame[frame], "backward_ms") for frame in frames}
    forward_by_frame = {str(frame): summary_median_ms(rows_by_frame[frame], "forward_ms") for frame in frames}
    requested_by_frame = {
        str(frame): int(rows_by_frame[frame].get("requested_frames", frame))
        for frame in frames
    }
    loaded_by_frame = {
        str(frame): int(rows_by_frame[frame].get("loaded_frame_count", frame))
        for frame in frames
    }
    repeat_used_by_frame = {
        str(frame): bool(rows_by_frame[frame].get("repeat_loaded_frames_used", False))
        for frame in frames
    }
    sample_count_by_frame = {}
    pair_count_by_frame = {}
    for frame in frames:
        row = rows_by_frame[frame]
        direct_count = summary_median_ms(row, "direct_grad_tube_count")
        sample_count = summary_median_ms(row, "sample_count")
        pair_count = summary_median_ms(row, "uvt_tile_tube_pairs")
        if direct_count is not None:
            sample_count_by_frame[str(frame)] = direct_count
        elif sample_count is not None:
            sample_count_by_frame[str(frame)] = sample_count
        if pair_count is not None:
            pair_count_by_frame[str(frame)] = pair_count
    first = str(frames[0]) if frames else None
    last = str(frames[-1]) if frames else None
    return {
        "status": "ok" if frames and all(row.get("sample_emission_mode") == "direct_atomic" for row in star_rows) else "failed",
        "frame_counts": frames,
        "requested_frame_count_by_frame": requested_by_frame,
        "loaded_frame_count_by_frame": loaded_by_frame,
        "repeat_loaded_frames_used_by_frame": repeat_used_by_frame,
        "total_median_ms_by_frame": total_by_frame,
        "backward_median_ms_by_frame": backward_by_frame,
        "forward_median_ms_by_frame": forward_by_frame,
        "direct_grad_tube_count_by_frame": sample_count_by_frame,
        "uvt_tile_tube_pairs_by_frame": pair_count_by_frame,
        "total_median_scale_first_to_last": scale(total_by_frame.get(first), total_by_frame.get(last)) if first and last else None,
        "backward_median_scale_first_to_last": scale(backward_by_frame.get(first), backward_by_frame.get(last)) if first and last else None,
        "forward_median_scale_first_to_last": scale(forward_by_frame.get(first), forward_by_frame.get(last)) if first and last else None,
    }


def summarize_worldfoam(payload: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    rows_by_frame = {int(row["frame_count"]): row for row in rows if isinstance(row.get("frame_count"), int)}
    frames = sorted(rows_by_frame)
    total_by_frame = {str(frame): worldfoam_median_ms(rows_by_frame[frame], "total") for frame in frames}
    backward_by_frame = {str(frame): worldfoam_median_ms(rows_by_frame[frame], "backward") for frame in frames}
    loaded_by_frame = {
        str(frame): int(rows_by_frame[frame].get("loaded_frame_count", frame))
        for frame in frames
    }
    repeat_by_frame = {
        str(frame): bool(rows_by_frame[frame].get("repeat_loaded_frames", False))
        for frame in frames
    }
    repeat_scope_by_frame = {
        str(frame): str(rows_by_frame[frame].get("repeat_loaded_frames_scope", "real loaded frame count"))
        for frame in frames
    }
    train_tape_by_frame = {str(frame): worldfoam_wall_s(rows_by_frame[frame], "build_train_tape_s") for frame in frames}
    heldout_tape_by_frame = {str(frame): worldfoam_wall_s(rows_by_frame[frame], "build_heldout_tape_s") for frame in frames}
    train_loop_by_frame = {str(frame): worldfoam_wall_s(rows_by_frame[frame], "train_loop_s") for frame in frames}
    candidate_csr = (
        payload.get("gate4_affine_candidate_csr_fused_mse") is True
        or payload.get("tape_mode") in GATE4_AFFINE_CANDIDATE_CSR_MODES
    )
    owner_run_factorized = payload.get("tape_mode") in OWNER_RUN_FACTORIZE_MODES
    legacy_fused_mse = payload.get("vjp_mode") == "fused_mse_rgb_only"
    mixed_storage_by_frame = {
        str(frame): _row_float(
            rows_by_frame[frame],
            "train_mixed_tape_storage_bytes",
            "train_selected_tape_mps_resident_noncoeff_storage_bytes",
            "train_selected_tape_storage_bytes",
        )
        for frame in frames
    }
    explicit_storage_by_frame = {
        str(frame): _row_float(rows_by_frame[frame], "train_explicit_ray_storage_bytes")
        or _worldfoam_affine_ray_storage(rows_by_frame[frame])
        for frame in frames
    }
    candidate_count_by_frame = {
        str(frame): float(rows_by_frame[frame].get("gate4_endpoint_train_metadata", {}).get("candidate_count", 0))
        for frame in frames
        if isinstance(rows_by_frame[frame].get("gate4_endpoint_train_metadata"), dict)
    }
    max_candidates_by_frame = {
        str(frame): float(rows_by_frame[frame].get("gate4_endpoint_train_metadata", {}).get("max_candidates_per_row", 0))
        for frame in frames
        if isinstance(rows_by_frame[frame].get("gate4_endpoint_train_metadata"), dict)
    }
    first = str(frames[0]) if frames else None
    last = str(frames[-1]) if frames else None
    return {
        "status": "ok"
        if payload.get("status") == "ok" and frames and (legacy_fused_mse or candidate_csr or owner_run_factorized)
        else "failed",
        "benchmark_environment_status": payload.get("benchmark_environment", {}).get("status")
        if isinstance(payload.get("benchmark_environment"), dict)
        else None,
        "vjp_mode": payload.get("vjp_mode"),
        "tape_mode": payload.get("tape_mode"),
        "worldfoam_family": (
            "owner_run_factorized"
            if owner_run_factorized
            else "gate4_affine_candidate_csr"
            if candidate_csr
            else "gate4_fused_mse"
        ),
        "frame_counts": frames,
        "loaded_frame_count_by_frame": loaded_by_frame,
        "repeat_loaded_frames_by_frame": repeat_by_frame,
        "repeat_loaded_frames_scope_by_frame": repeat_scope_by_frame,
        "total_median_ms_by_frame": total_by_frame,
        "backward_median_ms_by_frame": backward_by_frame,
        "train_tape_build_s_by_frame": train_tape_by_frame,
        "heldout_tape_build_s_by_frame": heldout_tape_by_frame,
        "train_loop_s_by_frame": train_loop_by_frame,
        "train_mixed_tape_storage_bytes_by_frame": mixed_storage_by_frame,
        "train_explicit_ray_storage_bytes_by_frame": explicit_storage_by_frame,
        "candidate_count_by_frame": candidate_count_by_frame,
        "max_candidates_per_row_by_frame": max_candidates_by_frame,
        "train_psnr_by_frame": {str(frame): float(rows_by_frame[frame]["final_train_psnr"]) for frame in frames},
        "heldout_psnr_by_frame": {str(frame): float(rows_by_frame[frame]["final_heldout_psnr"]) for frame in frames},
        "total_median_scale_first_to_last": scale(total_by_frame.get(first), total_by_frame.get(last)) if first and last else None,
        "backward_median_scale_first_to_last": scale(backward_by_frame.get(first), backward_by_frame.get(last)) if first and last else None,
        "train_mixed_tape_storage_scale_first_to_last": scale(
            mixed_storage_by_frame.get(first), mixed_storage_by_frame.get(last)
        )
        if first and last
        else None,
        "train_explicit_ray_storage_scale_first_to_last": scale(
            explicit_storage_by_frame.get(first), explicit_storage_by_frame.get(last)
        )
        if first and last
        else None,
        "candidate_count_scale_first_to_last": scale(candidate_count_by_frame.get(first), candidate_count_by_frame.get(last))
        if first and last and candidate_count_by_frame
        else None,
    }


def worldfoam_acceptance_failures(payload: dict[str, Any]) -> list[str]:
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict) or not acceptance:
        return ["WorldFoam artifact acceptance is missing"]
    failed_keys = [key for key, value in acceptance.items() if value is not True]
    if failed_keys:
        return [f"WorldFoam artifact acceptance failed: {','.join(sorted(failed_keys))}"]
    return []


def compare_summaries(star: dict[str, Any], worldfoam: dict[str, Any]) -> dict[str, Any]:
    frames = [frame for frame in star.get("frame_counts", []) if frame in set(worldfoam.get("frame_counts", []))]
    star_total = star.get("total_median_ms_by_frame", {})
    star_backward = star.get("backward_median_ms_by_frame", {})
    worldfoam_total = worldfoam.get("total_median_ms_by_frame", {})
    worldfoam_backward = worldfoam.get("backward_median_ms_by_frame", {})
    return {
        "frame_counts": frames,
        "total_median_ms_ratio_star_over_worldfoam_by_frame": {
            str(frame): scale(worldfoam_total.get(str(frame)), star_total.get(str(frame))) for frame in frames
        },
        "backward_median_ms_ratio_star_over_worldfoam_by_frame": {
            str(frame): scale(worldfoam_backward.get(str(frame)), star_backward.get(str(frame))) for frame in frames
        },
        "scale_ratios": {
            "star_total_median": star.get("total_median_scale_first_to_last"),
            "star_backward_median": star.get("backward_median_scale_first_to_last"),
            "worldfoam_total_median": worldfoam.get("total_median_scale_first_to_last"),
            "worldfoam_backward_median": worldfoam.get("backward_median_scale_first_to_last"),
            "worldfoam_mixed_tape_storage": worldfoam.get("train_mixed_tape_storage_scale_first_to_last"),
            "worldfoam_explicit_ray_storage": worldfoam.get("train_explicit_ray_storage_scale_first_to_last"),
        },
    }


def run_star_cases(
    *,
    video_path: Path,
    frame_counts: tuple[int, ...],
    target_size: int,
    tube_count: int,
    seed: int,
    spatial_precision: float,
    temporal_precision: float,
    opacity: float,
    tile_t: int,
    tile_capacity: int,
    lr: float,
    steps: int,
    warmup_steps: int,
    pair_count_every: int,
    repeat_loaded_frames: bool,
) -> list[dict[str, Any]]:
    if str(STAR_BENCHMARKS) not in sys.path:
        sys.path.insert(0, str(STAR_BENCHMARKS))
    from uvt_train_step_timing_probe import run_case  # noqa: PLC0415

    rows = []
    for frame_count in frame_counts:
        started_at = time.perf_counter()
        row = run_case(
            video_path=video_path,
            target_size=target_size,
            max_frames=frame_count,
            tube_count=tube_count,
            seed=seed,
            spatial_precision=spatial_precision,
            temporal_precision=temporal_precision,
            opacity=opacity,
            tile_t=tile_t,
            tile_capacity=tile_capacity,
            lr=lr,
            steps=steps,
            warmup_steps=warmup_steps,
            sample_count_every=0,
            pair_count_every=pair_count_every,
            reduction_mode="index_add",
            sample_emission_mode="direct_atomic",
            tile_load_reg_weight=0.0,
            tile_load_target=0.0,
            repeat_loaded_frames=repeat_loaded_frames,
        )
        row["case_wall_s"] = float(time.perf_counter() - started_at)
        rows.append(row)
    return rows


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    frame_counts = parse_int_list(args.frame_counts)
    benchmark_environment_start = wait_for_benchmark_environment_ok(
        timeout_s=float(args.wait_for_benchmark_environment_ok_timeout_s)
        if args.require_benchmark_environment_ok
        else 0.0,
        poll_s=float(args.wait_for_benchmark_environment_ok_poll_s),
    )
    if args.require_benchmark_environment_ok and benchmark_environment_blocks_promotion(benchmark_environment_start):
        start_status = str(benchmark_environment_start.get("status", "unchecked"))
        failure = (
            "benchmark environment was contended before STAR run"
            if start_status == "contended"
            else f"benchmark environment was not promotable before STAR run: {start_status}"
        )
        return {
            "benchmark": "star_uvt_vs_worldfoam_gate4_scale_mps",
            "status": "failed",
            "failures": [failure],
            "benchmark_environment": merge_benchmark_environments(
                benchmark_environment_start,
                benchmark_environment_start,
            ),
            "frame_counts": list(frame_counts),
            "steps": int(args.steps),
            "warmup_steps": int(args.warmup_steps),
            "worldfoam": {
                "artifact": str(args.worldfoam_artifact),
            },
        }
    worldfoam_payload = load_json(args.worldfoam_artifact)
    if args.require_clean_worldfoam_artifact:
        acceptance_failures = worldfoam_acceptance_failures(worldfoam_payload)
        if acceptance_failures:
            return {
                "benchmark": "star_uvt_vs_worldfoam_gate4_scale_mps",
                "status": "failed",
                "failures": acceptance_failures,
                "benchmark_environment": merge_benchmark_environments(
                    benchmark_environment_start,
                    benchmark_environment_start,
                ),
                "frame_counts": list(frame_counts),
                "steps": int(args.steps),
                "warmup_steps": int(args.warmup_steps),
                "worldfoam": {
                    "artifact": str(args.worldfoam_artifact),
                    "summary": summarize_worldfoam(worldfoam_payload),
                },
                "star": {"rows": [], "summary": {"status": "not_run"}},
            }
    star_rows = run_star_cases(
        video_path=args.video_path,
        frame_counts=frame_counts,
        target_size=args.star_target_size,
        tube_count=args.star_tube_count,
        seed=args.star_seed,
        spatial_precision=args.star_spatial_precision,
        temporal_precision=args.star_temporal_precision,
        opacity=args.star_opacity,
        tile_t=args.star_tile_t,
        tile_capacity=args.star_tile_capacity,
        lr=args.star_lr,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        pair_count_every=args.star_pair_count_every,
        repeat_loaded_frames=bool(args.star_repeat_loaded_frames),
    )
    benchmark_environment = merge_benchmark_environments_with_optional_settle(
        benchmark_environment_start,
        settle_s=float(args.post_run_benchmark_environment_settle_s),
    )
    star_summary = summarize_star_rows(star_rows)
    worldfoam_summary = summarize_worldfoam(worldfoam_payload)
    comparison = compare_summaries(star_summary, worldfoam_summary)
    failures = []
    if star_summary["frame_counts"] != list(frame_counts):
        failures.append("STAR frame counts did not match requested frame counts")
    if worldfoam_summary["frame_counts"] != list(frame_counts):
        failures.append("WorldFoam frame counts did not match requested frame counts")
    if star_summary["status"] != "ok":
        failures.append("STAR summary failed")
    if worldfoam_summary["status"] != "ok":
        failures.append("WorldFoam summary failed or artifact is not fused_mse_rgb_only/candidate_csr")
    if args.require_clean_worldfoam_artifact:
        worldfoam_environment_status = worldfoam_summary.get("benchmark_environment_status")
        if worldfoam_environment_status is None:
            failures.append("WorldFoam artifact has no benchmark_environment status")
        elif worldfoam_environment_status == "contended":
            failures.append("WorldFoam artifact benchmark_environment is contended")
        elif worldfoam_environment_status not in {"ok", "background"}:
            failures.append(f"WorldFoam artifact benchmark_environment is not promotable: {worldfoam_environment_status}")
    if args.require_benchmark_environment_ok and benchmark_environment_blocks_promotion(benchmark_environment):
        merged_status = str(benchmark_environment.get("status", "unchecked"))
        if merged_status == "contended":
            failures.append("benchmark environment became contended during STAR run")
        else:
            failures.append(f"benchmark environment was not promotable during STAR run: {merged_status}")
    return {
        "benchmark": "star_uvt_vs_worldfoam_gate4_scale_mps",
        "status": "failed" if failures else "ok",
        "failures": failures,
        "benchmark_environment": benchmark_environment,
        "scope": (
            "Matched small-MPS speed gate only: STAR UVT direct_atomic source-video timing "
            "versus WorldFoam moving-camera frozen-geometry train/eval. "
            "This is not a quality/capacity parity proof."
        ),
        "frame_counts": list(frame_counts),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "star": {
            "video_path": str(args.video_path),
            "target_size": int(args.star_target_size),
            "tube_count": int(args.star_tube_count),
            "repeat_loaded_frames": bool(args.star_repeat_loaded_frames),
            "sample_emission_mode": "direct_atomic",
            "reduction_mode": "index_add",
            "rows": star_rows,
            "summary": star_summary,
        },
        "worldfoam": {
            "artifact": str(args.worldfoam_artifact),
            "summary": worldfoam_summary,
        },
        "comparison": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a matched STAR-UVT vs WorldFoam Gate4 scale gate.")
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--worldfoam-artifact", type=Path, default=DEFAULT_WORLDFOAM_ARTIFACT)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--star-target-size", type=int, default=32)
    parser.add_argument("--star-tube-count", type=int, default=224)
    parser.add_argument("--star-seed", type=int, default=5)
    parser.add_argument("--star-spatial-precision", type=float, default=0.125)
    parser.add_argument("--star-temporal-precision", type=float, default=2.0)
    parser.add_argument("--star-opacity", type=float, default=0.7)
    parser.add_argument("--star-tile-t", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--star-tile-capacity", type=int, choices=(32, 64, 128, 256), default=128)
    parser.add_argument("--star-lr", type=float, default=0.12)
    parser.add_argument("--star-pair-count-every", type=int, default=0)
    parser.add_argument(
        "--star-repeat-loaded-frames",
        action="store_true",
        help=(
            "Pass --repeat-loaded-frames to the STAR timing probe. This is only for synthetic "
            "repeated-fixture speed scaling when requested frame counts exceed the real video length."
        ),
    )
    parser.add_argument(
        "--require-clean-worldfoam-artifact",
        action="store_true",
        help="Fail if the WorldFoam artifact lacks benchmark_environment status or ended contended.",
    )
    parser.add_argument(
        "--require-benchmark-environment-ok",
        action="store_true",
        help="Fail before or after the STAR run if benchmark_environment.status is contended.",
    )
    parser.add_argument(
        "--wait-for-benchmark-environment-ok-timeout-s",
        type=float,
        default=0.0,
        help="With --require-benchmark-environment-ok, poll until start is promotable or this timeout expires.",
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
            "When the immediate post-STAR blocker is only MTLCompilerService, wait this long and "
            "use a second environment snapshot for promotion. Other Python/Torch/MPS blockers "
            "still make the artifact diagnostic."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "star_uvt_vs_worldfoam_gate4_scale_mps.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_gate(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
