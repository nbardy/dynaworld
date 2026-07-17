#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from run_worldfoam_next_mps_candidate import DEFAULT_TAPE_MODE


EXPECTED_BENCHMARK = "world_foam_lane2_segment_tape_train_eval_mps"
EXPECTED_DEVICE = "mps"
EXPECTED_OPTIMIZER_MODE = "manual-vjp"
EXPECTED_ENDPOINT_RECORD_SOURCE = "slow-owner-run"
MIN_STABILITY_SAMPLES = 3
EXPECTED_BLOCKING_CPU_THRESHOLD = 5.0
EXPECTED_GENERAL_BLOCKING_CPU_THRESHOLD = 75.0
REQUIRED_FRAME_COUNTS = [2, 4, 8, 16, 32]
REQUIRED_RENDER_SIZE = 64
REQUIRED_SITE_COUNT = 24
REQUIRED_STEPS = 8
REQUIRED_WARMUP_STEPS = 4


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_path(value: Any, *, base_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _same_path(left: Any, right: Any, *, base_dir: Path) -> bool:
    left_path = _resolve_path(left, base_dir=base_dir)
    right_path = _resolve_path(right, base_dir=base_dir)
    if left_path is None or right_path is None:
        return False
    return left_path.resolve(strict=False) == right_path.resolve(strict=False)


def _command_value(command: Any, option: str) -> str | None:
    if not isinstance(command, list):
        return None
    for idx, item in enumerate(command[:-1]):
        if item == option and isinstance(command[idx + 1], str):
            return command[idx + 1]
    return None


def _parse_frame_counts(value: Any) -> list[int]:
    if isinstance(value, list) and value and all(isinstance(item, int) for item in value):
        return [int(item) for item in value]
    if not isinstance(value, str):
        return []
    out = []
    for raw_item in value.split(","):
        try:
            out.append(int(raw_item.strip()))
        except ValueError:
            return []
    return out


def _parse_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_float(value: Any) -> float | None:
    if not _is_number(value):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _is_clean_environment(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    return payload.get("status") in {"ok", "background"}


def _check_benchmark_environment_contract(
    environment: dict[str, Any] | None,
    *,
    label: str,
    failures: list[str],
) -> None:
    if not isinstance(environment, dict):
        failures.append(f"{label} benchmark environment is missing")
        return
    snapshots = []
    if isinstance(environment.get("start"), dict) or isinstance(environment.get("end"), dict):
        snapshots.extend(
            snapshot
            for snapshot in (environment.get("start"), environment.get("end"))
            if isinstance(snapshot, dict)
        )
    else:
        snapshots.append(environment)
    if not snapshots:
        failures.append(f"{label} benchmark environment has no snapshots")
        return
    for index, snapshot in enumerate(snapshots):
        suffix = f" snapshot {index}" if len(snapshots) > 1 else ""
        if snapshot.get("blocking_cpu_threshold") != EXPECTED_BLOCKING_CPU_THRESHOLD:
            failures.append(f"{label} benchmark environment{suffix} missing current blocking_cpu_threshold")
        if snapshot.get("general_blocking_cpu_threshold") != EXPECTED_GENERAL_BLOCKING_CPU_THRESHOLD:
            failures.append(
                f"{label} benchmark environment{suffix} missing current general_blocking_cpu_threshold"
            )
        if snapshot.get("blocking_process_count") != 0:
            failures.append(f"{label} benchmark environment{suffix} recorded blocking processes")
        if snapshot.get("contending_process_count") != 0:
            failures.append(f"{label} benchmark environment{suffix} recorded contending processes")
        keywords = snapshot.get("keywords")
        if not isinstance(keywords, list) or "python" not in keywords:
            failures.append(f"{label} benchmark environment{suffix} missing python keyword coverage")
        hard_keywords = snapshot.get("hard_keywords")
        if not isinstance(hard_keywords, list) or "torch" not in hard_keywords or "mps" not in hard_keywords:
            failures.append(f"{label} benchmark environment{suffix} missing torch/mps hard keyword coverage")


def _artifact_environment(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    environment = payload.get("benchmark_environment")
    return environment if isinstance(environment, dict) else None


def _row_step_mean(row: dict[str, Any], key: str) -> float | None:
    step_summary = row.get("step_summary")
    if not isinstance(step_summary, dict):
        return None
    section = step_summary.get(key)
    if not isinstance(section, dict):
        return None
    return _finite_float(section.get("mean_s"))


def _row_uses_fused_render_timing(row: dict[str, Any]) -> bool:
    render_mean = _row_step_mean(row, "render")
    fused_mean = _row_step_mean(row, "fused_loss_vjp")
    return (
        render_mean is not None
        and abs(render_mean) <= 1.0e-12
        and fused_mean is not None
        and fused_mean > 0.0
    )


def _artifact_uses_fused_render_timing(artifact: dict[str, Any]) -> bool:
    rows = artifact.get("rows")
    return isinstance(rows, list) and bool(rows) and all(
        isinstance(row, dict) and _row_uses_fused_render_timing(row) for row in rows
    )


def _scale_from_rows(artifact: dict[str, Any], key: str) -> float | None:
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return None
    first = rows[0]
    last = rows[-1]
    if not isinstance(first, dict) or not isinstance(last, dict):
        return None
    first_mean = _row_step_mean(first, key)
    last_mean = _row_step_mean(last, key)
    if first_mean is None or last_mean is None:
        return None
    if abs(first_mean) <= 1.0e-12:
        return 0.0 if abs(last_mean) <= 1.0e-12 else float("inf")
    return last_mean / first_mean


def _check_summary(summary: dict[str, Any] | None, failures: list[str]) -> None:
    if summary is None:
        failures.append("summary is missing or invalid JSON")
        return
    if summary.get("benchmark") != "world_foam_next_mps_candidate_launch":
        failures.append("summary benchmark is not world_foam_next_mps_candidate_launch")
    if summary.get("status") != "train_eval_ok":
        failures.append(f"summary status is not train_eval_ok: {summary.get('status')!r}")
    if summary.get("execute") is not True:
        failures.append("summary was not executed")
    if summary.get("train_eval_returncode") != 0:
        failures.append("summary train_eval_returncode is not 0")
    if summary.get("readiness_status") != "ok":
        failures.append("summary readiness_status is not ok")
    if summary.get("ready_for_quiet_mps_quality_speed_run") is not True:
        failures.append("summary readiness did not mark candidate ready")
    if not isinstance(summary.get("next_mps_candidate"), str):
        failures.append("summary is missing next_mps_candidate")
    requested = summary.get("preflight_stability_samples_requested")
    completed = summary.get("preflight_stability_samples_completed")
    if not isinstance(requested, int) or requested < MIN_STABILITY_SAMPLES:
        failures.append(f"preflight stability must request at least {MIN_STABILITY_SAMPLES} samples")
    if completed != requested:
        failures.append("preflight stability did not complete all requested samples")
    if summary.get("preflight_stability_ok") is not True:
        failures.append("preflight_stability_ok is not true")
    if summary.get("preflight_returncode") != 0:
        failures.append("preflight_returncode is not 0")
    if summary.get("preflight_benchmark_environment_status") not in {"ok", "background"}:
        failures.append("preflight benchmark environment is not clean")
    if summary.get("preflight_blocking_process_count") not in {0, None}:
        failures.append("preflight recorded blocking processes")
    if summary.get("preflight_contending_process_count") not in {0, None}:
        failures.append("preflight recorded contending processes")
    _check_benchmark_environment_contract(
        summary.get("preflight_benchmark_environment"),
        label="summary preflight",
        failures=failures,
    )


def _check_command_contract(
    summary: dict[str, Any],
    *,
    artifact_path: Path,
    expected_candidate: str,
    base_dir: Path,
    failures: list[str],
) -> list[int]:
    command = summary.get("train_eval_command")
    if not isinstance(command, list):
        failures.append("summary is missing train_eval_command")
        return []
    if "--require-benchmark-environment-ok" not in command:
        failures.append("train_eval_command does not require clean benchmark environment")
    if _command_value(command, "--optimizer-mode") != EXPECTED_OPTIMIZER_MODE:
        failures.append("train_eval_command optimizer mode is not manual-vjp")
    if _command_value(command, "--tape-mode") != DEFAULT_TAPE_MODE:
        failures.append("train_eval_command tape mode is not the expected native-cutwalk mode")
    if _command_value(command, "--endpoint-record-source") != EXPECTED_ENDPOINT_RECORD_SOURCE:
        failures.append("train_eval_command endpoint source is not slow-owner-run")
    if _command_value(command, "--site-initialization") != expected_candidate:
        failures.append("train_eval_command site initialization does not match candidate")
    if not _same_path(_command_value(command, "--out-json"), str(artifact_path), base_dir=base_dir):
        failures.append("train_eval_command --out-json does not match planned artifact")
    frame_counts = _parse_frame_counts(_command_value(command, "--frame-counts"))
    if not frame_counts:
        failures.append("train_eval_command is missing parseable --frame-counts")
    elif frame_counts != REQUIRED_FRAME_COUNTS:
        failures.append(
            f"train_eval_command frame_counts {frame_counts} do not match required {REQUIRED_FRAME_COUNTS}"
        )
    required_int_options = (
        ("--render-size", REQUIRED_RENDER_SIZE),
        ("--site-count", REQUIRED_SITE_COUNT),
        ("--steps", REQUIRED_STEPS),
        ("--warmup-steps", REQUIRED_WARMUP_STEPS),
    )
    for option, expected in required_int_options:
        actual = _parse_int(_command_value(command, option))
        if actual != expected:
            failures.append(f"train_eval_command {option} {actual!r} does not match required {expected}")
    return frame_counts


def _check_artifact_contract(
    artifact: dict[str, Any] | None,
    *,
    expected_candidate: str,
    expected_frame_counts: list[int],
    failures: list[str],
) -> None:
    if artifact is None:
        failures.append("WorldFoam artifact is missing or invalid JSON")
        return
    if artifact.get("status") != "ok":
        failures.append(f"WorldFoam artifact status is not ok: {artifact.get('status')!r}")
    if artifact.get("benchmark") != EXPECTED_BENCHMARK:
        failures.append("WorldFoam artifact benchmark is not the train/eval MPS benchmark")
    if artifact.get("device") != EXPECTED_DEVICE:
        failures.append("WorldFoam artifact device is not mps")
    for key, expected in (("render_size", REQUIRED_RENDER_SIZE), ("site_count", REQUIRED_SITE_COUNT)):
        if artifact.get(key) != expected:
            failures.append(f"WorldFoam artifact {key} {artifact.get(key)!r} does not match required {expected}")
    artifact_environment = _artifact_environment(artifact)
    if not _is_clean_environment(artifact_environment):
        failures.append("WorldFoam artifact benchmark_environment is not clean")
    _check_benchmark_environment_contract(
        artifact_environment,
        label="WorldFoam artifact",
        failures=failures,
    )
    if artifact.get("site_initialization") != expected_candidate:
        failures.append("WorldFoam artifact site_initialization does not match candidate")
    if artifact.get("tape_mode") != DEFAULT_TAPE_MODE:
        failures.append("WorldFoam artifact tape_mode is not the expected native-cutwalk mode")
    if artifact.get("optimizer_mode") != EXPECTED_OPTIMIZER_MODE:
        failures.append("WorldFoam artifact optimizer_mode is not manual-vjp")
    if artifact.get("endpoint_record_source") != EXPECTED_ENDPOINT_RECORD_SOURCE:
        failures.append("WorldFoam artifact endpoint_record_source is not slow-owner-run")
    if artifact.get("experimental_selected_only_owner_run_delta_prep") is not True:
        failures.append("WorldFoam artifact did not use selected-only owner-run delta prep")
    if artifact.get("experimental_native_owner_run_cutwalk_delta") is not True:
        failures.append("WorldFoam artifact did not use native owner-run cutwalk delta")
    if artifact.get("allow_repeat_loaded_frames") is True or artifact.get("repeat_loaded_frames") is True:
        failures.append("WorldFoam artifact used repeated loaded frames")
    artifact_frames = _parse_frame_counts(artifact.get("frame_counts"))
    expected = sorted(set(expected_frame_counts))
    if artifact_frames != expected:
        failures.append(f"WorldFoam artifact frame_counts {artifact_frames} do not match {expected}")
    _check_acceptance(artifact, failures)
    _check_rows(artifact, expected_frame_counts=expected, failures=failures)


def _check_acceptance(artifact: dict[str, Any], failures: list[str]) -> None:
    acceptance = artifact.get("acceptance")
    if not isinstance(acceptance, dict) or not acceptance:
        failures.append("WorldFoam artifact acceptance is missing")
        return
    failed = sorted(str(key) for key, value in acceptance.items() if value is not True)
    if failed:
        failures.append(f"WorldFoam artifact acceptance failed: {','.join(failed)}")
    for key in ("total_step_sublinear_vs_frames", "render_sublinear_vs_frames", "backward_sublinear_vs_frames"):
        if acceptance.get(key) is not True:
            failures.append(f"WorldFoam artifact missing required acceptance {key}=true")
    frame_scale = artifact.get("frame_scale_first_to_last")
    frame_scale_value = _finite_float(frame_scale)
    if frame_scale_value is None or frame_scale_value <= 0.0:
        failures.append("WorldFoam artifact frame_scale_first_to_last is not finite positive")
    uses_fused_render_timing = _artifact_uses_fused_render_timing(artifact)
    required_scale_keys = ["total_step_scale_first_to_last", "backward_scale_first_to_last"]
    if not uses_fused_render_timing:
        required_scale_keys.append("render_scale_first_to_last")
    for key in required_scale_keys:
        scale = artifact.get(key)
        scale_value = _finite_float(scale)
        if scale_value is None or scale_value <= 0.0:
            failures.append(f"WorldFoam artifact {key} is not finite positive")
        elif frame_scale_value is not None and scale_value >= frame_scale_value:
            failures.append(f"{key} is not sublinear versus frame scale")
    if uses_fused_render_timing:
        fused_scale = _scale_from_rows(artifact, "fused_loss_vjp")
        fused_scale_value = _finite_float(fused_scale)
        if fused_scale_value is None or fused_scale_value <= 0.0:
            failures.append("WorldFoam artifact fused_loss_vjp row scale is not finite positive")
        elif frame_scale_value is not None and fused_scale_value >= frame_scale_value:
            failures.append("fused_loss_vjp row scale is not sublinear versus frame scale")


def _check_rows(
    artifact: dict[str, Any],
    *,
    expected_frame_counts: list[int],
    failures: list[str],
) -> None:
    rows = artifact.get("rows")
    if not isinstance(rows, list) or not rows:
        failures.append("WorldFoam artifact has no rows")
        return
    row_by_frame = {}
    duplicate_frame_counts: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            failures.append("WorldFoam artifact row is not an object")
            continue
        frame_count = row.get("frame_count")
        if isinstance(frame_count, int) and not isinstance(frame_count, bool):
            if frame_count in row_by_frame:
                duplicate_frame_counts.add(frame_count)
            row_by_frame[frame_count] = row
        else:
            failures.append(f"WorldFoam artifact row has invalid frame_count {frame_count!r}")
    if len(rows) != len(expected_frame_counts):
        failures.append(f"WorldFoam row count {len(rows)} does not match expected {len(expected_frame_counts)}")
    if duplicate_frame_counts:
        failures.append(f"WorldFoam duplicate row frame counts {sorted(duplicate_frame_counts)}")
    if sorted(row_by_frame) != sorted(set(expected_frame_counts)):
        failures.append(f"WorldFoam row frame counts {sorted(row_by_frame)} do not match expected")
    for frame_count, row in sorted(row_by_frame.items()):
        if row.get("status") != "ok":
            failures.append(f"WorldFoam row {frame_count}f status is not ok")
        loaded = row.get("loaded_frame_count")
        if not isinstance(loaded, int) or loaded < frame_count:
            failures.append(f"WorldFoam row {frame_count}f loaded_frame_count is insufficient")
        if row.get("repeat_loaded_frames") is True:
            failures.append(f"WorldFoam row {frame_count}f used repeated loaded frames")
        for key, expected in (
            ("render_size", REQUIRED_RENDER_SIZE),
            ("site_count", REQUIRED_SITE_COUNT),
            ("steps", REQUIRED_STEPS),
            ("warmup_steps", REQUIRED_WARMUP_STEPS),
        ):
            if row.get(key) != expected:
                failures.append(f"WorldFoam row {frame_count}f {key} {row.get(key)!r} does not match required {expected}")
        for key in ("final_train_psnr", "final_heldout_psnr", "final_train_l1", "final_heldout_l1"):
            if _finite_float(row.get(key)) is None:
                failures.append(f"WorldFoam row {frame_count}f missing finite numeric {key}")
        for key in ("total", "backward"):
            mean_s = _row_step_mean(row, key)
            if mean_s is None or mean_s <= 0.0:
                failures.append(f"WorldFoam row {frame_count}f missing positive {key} mean_s")
        render_mean_s = _row_step_mean(row, "render")
        if render_mean_s is None or render_mean_s <= 0.0:
            fused_mean_s = _row_step_mean(row, "fused_loss_vjp")
            if fused_mean_s is None or fused_mean_s <= 0.0:
                failures.append(f"WorldFoam row {frame_count}f missing positive render mean_s")


def _brief_process(process: Any) -> dict[str, Any]:
    if not isinstance(process, dict):
        return {"raw": str(process)[:1024]}
    return {
        key: (str(value)[:1024] if key == "command" else value)
        for key, value in process.items()
        if key in {"pid", "ppid", "stat", "elapsed", "block_reason", "pcpu", "pmem", "command"}
    }


def verify_summary(summary_path: Path) -> dict[str, Any]:
    summary = _load_json(summary_path)
    failures: list[str] = []
    _check_summary(summary, failures)
    base_dir = summary_path.parent
    artifact_path = None
    artifact = None
    expected_frame_counts: list[int] = []
    artifact_checks_skipped = False
    if isinstance(summary, dict):
        artifact_path = _resolve_path(summary.get("planned_worldfoam_artifact"), base_dir=base_dir)
        candidate = summary.get("next_mps_candidate")
        expected_candidate = candidate if isinstance(candidate, str) else ""
        if artifact_path is None:
            failures.append("summary is missing planned_worldfoam_artifact")
        else:
            expected_frame_counts = _check_command_contract(
                summary,
                artifact_path=artifact_path,
                expected_candidate=expected_candidate,
                base_dir=base_dir,
                failures=failures,
            )
            if summary.get("status") == "train_eval_ok" and summary.get("train_eval_returncode") == 0:
                artifact = _load_json(artifact_path)
                _check_artifact_contract(
                    artifact,
                    expected_candidate=expected_candidate,
                    expected_frame_counts=expected_frame_counts,
                    failures=failures,
                )
            else:
                artifact_checks_skipped = True
    preflight_blocking_processes = summary.get("preflight_blocking_processes") if isinstance(summary, dict) else None
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "summary_path": str(summary_path),
        "worldfoam_artifact": str(artifact_path) if artifact_path is not None else None,
        "expected_frame_counts": expected_frame_counts,
        "artifact_checks_skipped": artifact_checks_skipped,
        "preflight_benchmark_environment_status": summary.get("preflight_benchmark_environment_status")
        if isinstance(summary, dict)
        else None,
        "preflight_blocking_process_count": summary.get("preflight_blocking_process_count")
        if isinstance(summary, dict)
        else None,
        "preflight_contending_process_count": summary.get("preflight_contending_process_count")
        if isinstance(summary, dict)
        else None,
        "preflight_blocking_reasons": summary.get("preflight_blocking_reasons")
        if isinstance(summary, dict)
        else None,
        "preflight_external_blocker_summary": summary.get("preflight_external_blocker_summary")
        if isinstance(summary, dict)
        else None,
        "preflight_blocking_processes": [_brief_process(process) for process in preflight_blocking_processes]
        if isinstance(preflight_blocking_processes, list)
        else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the readiness-selected WorldFoam MPS quality/speed result."
    )
    parser.add_argument("summary_json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_summary(args.summary_json)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
