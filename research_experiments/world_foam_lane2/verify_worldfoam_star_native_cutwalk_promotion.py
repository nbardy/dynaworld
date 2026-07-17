#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_worldfoam_star_native_cutwalk_gate import DEFAULT_TAPE_MODE, SUMMARY_SCHEMA_VERSION


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


def _env_status(payload: dict[str, Any] | None) -> str | None:
    environment = payload.get("benchmark_environment") if isinstance(payload, dict) else None
    return environment.get("status") if isinstance(environment, dict) else None


def _is_clean_environment(status: Any) -> bool:
    return status in {"ok", "background"}


def _same_path(left: Any, right: Any, *, base_dir: Path) -> bool:
    left_path = _resolve_path(left, base_dir=base_dir)
    right_path = _resolve_path(right, base_dir=base_dir)
    if left_path is None or right_path is None:
        return False
    return left_path.resolve(strict=False) == right_path.resolve(strict=False)


def _command_has_path_option(command: Any, option: str, expected: Any, *, base_dir: Path) -> bool:
    if not isinstance(command, list) or not isinstance(expected, str) or not expected:
        return False
    for idx, item in enumerate(command[:-1]):
        if item != option:
            continue
        value = command[idx + 1]
        if value == expected or _same_path(value, expected, base_dir=base_dir):
            return True
    return False


def _same_recorded_path(left: Any, right: Any, *, base_dir: Path) -> bool:
    return isinstance(left, str) and isinstance(right, str) and bool(left) and bool(right) and (
        left == right or _same_path(left, right, base_dir=base_dir)
    )


def _frame_counts_from_payload(payload: dict[str, Any] | None) -> list[int]:
    if not isinstance(payload, dict):
        return []
    frames = payload.get("frame_counts")
    if isinstance(frames, list) and all(isinstance(item, int) for item in frames):
        return list(frames)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("frame_count"), int):
            out.append(int(row["frame_count"]))
    return sorted(set(out))


def _summary_frame_counts(summary: dict[str, Any]) -> list[int]:
    frame_counts = summary.get("frame_counts")
    if not isinstance(frame_counts, list) or not frame_counts:
        return []
    out = []
    for item in frame_counts:
        value = _int_value(item)
        if value is None:
            return []
        out.append(value)
    return out


def _star_frame_counts_from_payload(payload: dict[str, Any] | None) -> list[int]:
    if not isinstance(payload, dict):
        return []
    star = payload.get("star")
    rows = star.get("rows") if isinstance(star, dict) else None
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        requested = _int_value(row.get("requested_frames"))
        if requested is None:
            requested = _int_value(row.get("frames"))
        if requested is not None:
            out.append(requested)
    return sorted(set(out))


def _int_value(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _check_worldfoam_real_loaded_frames(payload: dict[str, Any] | None, failures: list[str]) -> None:
    if payload is None:
        return
    if payload.get("allow_repeat_loaded_frames") is True or payload.get("repeat_loaded_frames") is True:
        failures.append("WorldFoam artifact is marked as a repeated-loaded-frame smoke")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        failures.append("WorldFoam artifact has no rows for real-loaded-frame verification")
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        frame_count = _int_value(row.get("frame_count"))
        loaded_frame_count = _int_value(row.get("loaded_frame_count"))
        if frame_count is None or loaded_frame_count is None:
            failures.append("WorldFoam row is missing frame_count or loaded_frame_count metadata")
            continue
        if loaded_frame_count < frame_count:
            failures.append(
                f"WorldFoam row used too few loaded frames: frame_count={frame_count} "
                f"loaded_frame_count={loaded_frame_count}"
            )
        if row.get("repeat_loaded_frames") is True:
            failures.append(f"WorldFoam row {frame_count}f used repeated loaded frames")


def _check_star_real_loaded_frames(payload: dict[str, Any] | None, failures: list[str]) -> None:
    if payload is None:
        return
    star = payload.get("star")
    if not isinstance(star, dict):
        failures.append("STAR artifact is missing star section for real-loaded-frame verification")
        return
    if star.get("repeat_loaded_frames") is True:
        failures.append("STAR artifact is marked as a repeated-loaded-frame smoke")
    rows = star.get("rows")
    if not isinstance(rows, list) or not rows:
        failures.append("STAR artifact has no rows for real-loaded-frame verification")
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        requested = _int_value(row.get("requested_frames"))
        if requested is None:
            requested = _int_value(row.get("frames"))
        loaded_frame_count = _int_value(row.get("loaded_frame_count"))
        if requested is None or loaded_frame_count is None:
            failures.append("STAR row is missing requested frame or loaded_frame_count metadata")
            continue
        if loaded_frame_count < requested:
            failures.append(
                f"STAR row used too few loaded frames: requested={requested} "
                f"loaded_frame_count={loaded_frame_count}"
            )
        if row.get("repeat_loaded_frames") is True or row.get("repeat_loaded_frames_used") is True:
            failures.append(f"STAR row {requested}f used repeated loaded frames")


def _check_real_frame_input_contract(
    summary: dict[str, Any],
    *,
    worldfoam_payload: dict[str, Any] | None,
    star_payload: dict[str, Any] | None,
    summary_dir: Path,
    failures: list[str],
) -> None:
    worldfoam_config = summary.get("worldfoam_config")
    star_video_path = summary.get("star_video_path")
    if not isinstance(worldfoam_config, str) or not worldfoam_config:
        failures.append("real-loaded-frame promotion must record worldfoam_config")
    if not isinstance(star_video_path, str) or not star_video_path:
        failures.append("real-loaded-frame promotion must record star_video_path")
    if isinstance(worldfoam_config, str) and worldfoam_config:
        for command_key in ("worldfoam_preflight_command", "worldfoam_command"):
            if not _command_has_path_option(summary.get(command_key), "--config", worldfoam_config, base_dir=summary_dir):
                failures.append(f"{command_key} must pass --config matching worldfoam_config")
    if isinstance(star_video_path, str) and star_video_path:
        for command_key in ("planned_star_compare_command", "star_compare_command"):
            if not _command_has_path_option(summary.get(command_key), "--video-path", star_video_path, base_dir=summary_dir):
                failures.append(f"{command_key} must pass --video-path matching star_video_path")
    if isinstance(worldfoam_config, str) and worldfoam_config and isinstance(worldfoam_payload, dict):
        if not _same_recorded_path(worldfoam_payload.get("config_path"), worldfoam_config, base_dir=summary_dir):
            failures.append("WorldFoam artifact config_path must match worldfoam_config")
    if isinstance(star_video_path, str) and star_video_path and isinstance(star_payload, dict):
        star_section = star_payload.get("star")
        artifact_video_path = star_section.get("video_path") if isinstance(star_section, dict) else None
        if not _same_recorded_path(artifact_video_path, star_video_path, base_dir=summary_dir):
            failures.append("STAR artifact video_path must match star_video_path")


def _check_real_frame_count_contract(
    summary: dict[str, Any],
    *,
    worldfoam_payload: dict[str, Any] | None,
    star_payload: dict[str, Any] | None,
    failures: list[str],
) -> None:
    frame_counts = _summary_frame_counts(summary)
    if not frame_counts:
        failures.append("real-loaded-frame promotion must record frame_counts")
        return
    expected = sorted(set(frame_counts))
    worldfoam_frames = _frame_counts_from_payload(worldfoam_payload)
    star_frames = _star_frame_counts_from_payload(star_payload)
    if worldfoam_frames != expected:
        failures.append(f"WorldFoam artifact frame_counts {worldfoam_frames} do not match requested {expected}")
    if star_frames != expected:
        failures.append(f"STAR artifact frame_counts {star_frames} do not match requested {expected}")


def _check_worldfoam_payload(payload: dict[str, Any] | None, failures: list[str]) -> None:
    if payload is None:
        failures.append("WorldFoam artifact is missing or invalid JSON")
        return
    if payload.get("status") != "ok":
        failures.append("WorldFoam artifact status is not ok")
    environment_status = _env_status(payload)
    if not _is_clean_environment(environment_status):
        failures.append(f"WorldFoam artifact benchmark_environment is not clean: {environment_status}")
    if payload.get("tape_mode") != DEFAULT_TAPE_MODE:
        failures.append("WorldFoam artifact tape_mode is not the native-cutwalk framebitmask mode")
    if payload.get("endpoint_record_source") != "slow-owner-run":
        failures.append("WorldFoam artifact endpoint_record_source is not slow-owner-run")
    if payload.get("experimental_selected_only_owner_run_delta_prep") is not True:
        failures.append("WorldFoam artifact did not use selected-only owner-run delta prep")
    if payload.get("experimental_native_owner_run_cutwalk_delta") is not True:
        failures.append("WorldFoam artifact did not use native owner-run cutwalk delta")
    if not _frame_counts_from_payload(payload):
        failures.append("WorldFoam artifact has no frame-count rows")
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict) or not acceptance:
        failures.append("WorldFoam artifact acceptance is missing")
    else:
        failed_keys = [key for key, value in acceptance.items() if value is not True]
        if failed_keys:
            failures.append(f"WorldFoam artifact acceptance failed: {','.join(sorted(failed_keys))}")


def _check_star_payload(
    payload: dict[str, Any] | None,
    *,
    selected_worldfoam_artifact: Any,
    summary_dir: Path,
    failures: list[str],
) -> None:
    if payload is None:
        failures.append("STAR comparison artifact is missing or invalid JSON")
        return
    if payload.get("status") != "ok":
        failures.append("STAR comparison status is not ok")
    if payload.get("failures") not in ([], None):
        failures.append("STAR comparison artifact reports failures")
    environment_status = _env_status(payload)
    if not _is_clean_environment(environment_status):
        failures.append(f"STAR comparison benchmark_environment is not clean: {environment_status}")
    star_summary = payload.get("star", {}).get("summary") if isinstance(payload.get("star"), dict) else None
    if not isinstance(star_summary, dict) or star_summary.get("status") != "ok":
        failures.append("STAR summary is not ok")
    worldfoam_section = payload.get("worldfoam") if isinstance(payload.get("worldfoam"), dict) else None
    if not isinstance(worldfoam_section, dict):
        failures.append("STAR comparison artifact is missing worldfoam section")
        return
    if not _same_path(worldfoam_section.get("artifact"), selected_worldfoam_artifact, base_dir=summary_dir):
        failures.append("STAR comparison did not consume the selected WorldFoam artifact")
    worldfoam_summary = worldfoam_section.get("summary")
    if not isinstance(worldfoam_summary, dict) or worldfoam_summary.get("status") != "ok":
        failures.append("STAR comparison WorldFoam summary is not ok")
    elif not _is_clean_environment(worldfoam_summary.get("benchmark_environment_status")):
        failures.append("STAR comparison WorldFoam summary benchmark environment is not clean")


def verify_summary(summary_path: Path) -> dict[str, Any]:
    summary_path = summary_path.resolve(strict=False)
    summary = _load_json(summary_path)
    failures: list[str] = []
    if summary is None:
        return {
            "status": "failed",
            "summary_path": str(summary_path),
            "failures": ["promotion summary is missing or invalid JSON"],
        }

    summary_dir = summary_path.parent
    if summary.get("summary_schema_version") != SUMMARY_SCHEMA_VERSION:
        failures.append("promotion summary schema version is not current")
    if summary.get("status") != "ok":
        failures.append(f"promotion summary status is not ok: {summary.get('status')}")
    if summary.get("worldfoam_returncode") != 0:
        failures.append("WorldFoam returncode is not 0")
    if summary.get("worldfoam_status") != "ok":
        failures.append("WorldFoam summary status is not ok")
    if not _is_clean_environment(summary.get("worldfoam_benchmark_environment_status")):
        failures.append("WorldFoam summary benchmark_environment_status is not clean")
    if summary.get("star_compare_returncode") != 0:
        failures.append("STAR comparison returncode is not 0")
    if summary.get("star_compare_status") != "ok":
        failures.append("STAR comparison summary status is not ok")
    if not _is_clean_environment(summary.get("star_compare_benchmark_environment_status")):
        failures.append("STAR comparison summary benchmark_environment_status is not clean")

    selected_worldfoam = summary.get("worldfoam_artifact")
    selected_star = summary.get("star_compare_artifact")
    if not isinstance(selected_worldfoam, str) or not selected_worldfoam:
        failures.append("worldfoam_artifact is not selected")
    if not isinstance(selected_star, str) or not selected_star:
        failures.append("star_compare_artifact is not selected")
    if not _same_path(summary.get("worldfoam_promotable_artifact"), selected_worldfoam, base_dir=summary_dir):
        failures.append("worldfoam_promotable_artifact does not match worldfoam_artifact")
    if not _same_path(summary.get("worldfoam_latest_written_artifact"), selected_worldfoam, base_dir=summary_dir):
        failures.append("worldfoam_latest_written_artifact does not match selected artifact")
    if not _same_path(summary.get("star_compare_latest_attempt_artifact"), selected_star, base_dir=summary_dir):
        failures.append("star_compare_latest_attempt_artifact does not match selected artifact")
    if not _same_path(summary.get("star_compare_latest_written_artifact"), selected_star, base_dir=summary_dir):
        failures.append("star_compare_latest_written_artifact does not match selected artifact")
    if not isinstance(summary.get("planned_star_compare_command"), list) or not summary["planned_star_compare_command"]:
        failures.append("planned_star_compare_command is not selected")
    elif isinstance(selected_worldfoam, str) and selected_worldfoam:
        if not _command_has_path_option(
            summary.get("planned_star_compare_command"),
            "--worldfoam-artifact",
            selected_worldfoam,
            base_dir=summary_dir,
        ):
            failures.append("planned_star_compare_command must consume the selected WorldFoam artifact")
    if not isinstance(summary.get("star_compare_command"), list) or not summary["star_compare_command"]:
        failures.append("star_compare_command is not selected")
    elif isinstance(selected_worldfoam, str) and selected_worldfoam:
        if not _command_has_path_option(
            summary.get("star_compare_command"),
            "--worldfoam-artifact",
            selected_worldfoam,
            base_dir=summary_dir,
        ):
            failures.append("star_compare_command must consume the selected WorldFoam artifact")

    attempts = summary.get("worldfoam_attempts")
    if not isinstance(attempts, list) or not attempts:
        failures.append("promotion summary has no WorldFoam attempts")
    else:
        promotable_attempts = [attempt for attempt in attempts if isinstance(attempt, dict) and attempt.get("promotable")]
        if len(promotable_attempts) != 1:
            failures.append("promotion summary must contain exactly one promotable WorldFoam attempt")
        elif not _same_path(promotable_attempts[0].get("artifact"), selected_worldfoam, base_dir=summary_dir):
            failures.append("promotable WorldFoam attempt does not match selected artifact")
        for attempt in promotable_attempts:
            if attempt.get("artifact_written") is not True:
                failures.append("promotable WorldFoam attempt did not write an artifact")
            if attempt.get("preflight_returncode") != 0:
                failures.append("promotable WorldFoam attempt preflight returncode is not 0")
            if not _is_clean_environment(attempt.get("preflight_benchmark_environment_status")):
                failures.append("promotable WorldFoam attempt preflight environment is not clean")
            if attempt.get("returncode") != 0:
                failures.append("promotable WorldFoam attempt returncode is not 0")
            if attempt.get("status") != "ok":
                failures.append("promotable WorldFoam attempt status is not ok")
            if not _is_clean_environment(attempt.get("benchmark_environment_status")):
                failures.append("promotable WorldFoam attempt benchmark environment is not clean")

    star_attempts = summary.get("star_compare_attempts")
    if isinstance(star_attempts, list) and star_attempts:
        promotable_star_attempts = [
            attempt for attempt in star_attempts if isinstance(attempt, dict) and attempt.get("promotable")
        ]
        if len(promotable_star_attempts) != 1:
            failures.append("promotion summary must contain exactly one promotable STAR attempt")
        elif not _same_path(promotable_star_attempts[0].get("artifact"), selected_star, base_dir=summary_dir):
            failures.append("promotable STAR attempt does not match selected STAR artifact")
        for attempt in promotable_star_attempts:
            if attempt.get("artifact_written") is not True:
                failures.append("promotable STAR attempt did not write an artifact")
            if attempt.get("returncode") != 0:
                failures.append("promotable STAR attempt returncode is not 0")
            if attempt.get("status") != "ok":
                failures.append("promotable STAR attempt status is not ok")
            if not _is_clean_environment(attempt.get("benchmark_environment_status")):
                failures.append("promotable STAR attempt benchmark environment is not clean")
    else:
        failures.append("promotion summary has no STAR attempts")

    worldfoam_path = _resolve_path(selected_worldfoam, base_dir=summary_dir)
    star_path = _resolve_path(summary.get("star_compare_artifact"), base_dir=summary_dir)
    worldfoam_payload = _load_json(worldfoam_path) if worldfoam_path else None
    star_payload = _load_json(star_path) if star_path else None
    _check_worldfoam_payload(worldfoam_payload, failures)
    _check_star_payload(
        star_payload,
        selected_worldfoam_artifact=selected_worldfoam,
        summary_dir=summary_dir,
        failures=failures,
    )
    if summary.get("require_real_loaded_frames") is True:
        if summary.get("repeat_loaded_frames") is True:
            failures.append("promotion summary requested repeated loaded frames while requiring real loaded frames")
        _check_real_frame_input_contract(
            summary,
            worldfoam_payload=worldfoam_payload,
            star_payload=star_payload,
            summary_dir=summary_dir,
            failures=failures,
        )
        _check_real_frame_count_contract(
            summary,
            worldfoam_payload=worldfoam_payload,
            star_payload=star_payload,
            failures=failures,
        )
        _check_worldfoam_real_loaded_frames(worldfoam_payload, failures)
        _check_star_real_loaded_frames(star_payload, failures)

    return {
        "status": "failed" if failures else "ok",
        "summary_path": str(summary_path),
        "worldfoam_artifact": selected_worldfoam,
        "star_compare_artifact": summary.get("star_compare_artifact"),
        "failures": failures,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a clean WorldFoam native-cutwalk to STAR promotion summary.")
    parser.add_argument("summary_json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify_summary(args.summary_json)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
