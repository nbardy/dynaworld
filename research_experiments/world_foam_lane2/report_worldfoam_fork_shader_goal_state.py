#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Any

from run_worldfoam_next_mps_candidate import _external_blocker_summary
from verify_worldfoam_next_mps_candidate_result import verify_summary as verify_next_mps_summary
from verify_worldfoam_native_variant_sources import DEFAULT_VARIANTS
from verify_worldfoam_rebuilt_native_smokes import REQUIRED_ARTIFACTS


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research_experiments" / "world_foam_lane2" / "results"
DEFAULT_SOURCE_JSON = RESULTS_DIR / "2026-05-21_worldfoam_native_variant_source_wiring.json"
DEFAULT_IMPORT_JSON = RESULTS_DIR / "2026-05-21_worldfoam_native_variant_import_registration.json"
DEFAULT_SMOKE_BUNDLE_JSON = RESULTS_DIR / "2026-05-21_worldfoam_rebuilt_native_smoke_bundle_verifier.json"
DEFAULT_NEXT_MPS_SUMMARY_JSON = (
    RESULTS_DIR / "2026-05-21_worldfoam_next_mps_goal_continuation_clean_retry.launch_summary.json"
)
DEFAULT_BLOCKER_DIAGNOSIS_JSON = RESULTS_DIR / "2026-05-21_worldfoam_mps_blocker_diagnosis.json"
NEXT_MPS_SUMMARY_GLOB = "*worldfoam_next_mps*.launch_summary.json"
BLOCKED_NEXT_MPS_STATUSES = frozenset({"preflight_contended", "preflight_retry_waiting"})
DEFAULT_MAX_BLOCKER_DIAGNOSIS_AGE_S = 900.0
CLEAN_MPS_RERUN_COMMAND = [
    "rtk",
    "env",
    "PYTHONPATH=research_experiments/world_foam_lane2:src/train",
    "PYTHONDONTWRITEBYTECODE=1",
    ".venv/bin/python",
    "research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py",
    "--execute",
    "--verify-result",
    "--preflight-stability-samples",
    "3",
    "--preflight-stability-interval-s",
    "5",
    "--preflight-retry-timeout-s",
    "900",
    "--preflight-retry-poll-s",
    "30",
]
CLEAN_MPS_RESULT_VERIFIER_COMMAND_TEMPLATE = [
    "rtk",
    "env",
    "PYTHONPATH=research_experiments/world_foam_lane2:src/train",
    "PYTHONDONTWRITEBYTECODE=1",
    ".venv/bin/python",
    "research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py",
    "<launch_summary_json>",
]


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "JSON root is not an object"
    return payload, None


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


def _nonempty_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _path_exists(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).exists()


def _variant_bundle_failures(payload: dict[str, Any], *, mode: str) -> list[str]:
    failures = []
    expected = {variant: package for variant, package in DEFAULT_VARIANTS}
    if not _path_exists(payload.get("variant_root")):
        failures.append("variant_root is missing or does not exist")
    variants = payload.get("variants")
    if payload.get("variant_count") != len(DEFAULT_VARIANTS):
        failures.append(f"variant_count is {payload.get('variant_count')!r}, expected {len(DEFAULT_VARIANTS)}")
    if not isinstance(variants, list):
        failures.append("variants is missing or not a list")
        return failures
    if len(variants) != len(DEFAULT_VARIANTS):
        failures.append(f"variants length is {len(variants)}, expected {len(DEFAULT_VARIANTS)}")
    seen: set[str] = set()
    for row in variants:
        if not isinstance(row, dict):
            failures.append("variant row is not an object")
            continue
        variant = row.get("variant")
        if not isinstance(variant, str):
            failures.append("variant row has invalid variant name")
            continue
        seen.add(variant)
        if row.get("package") != expected.get(variant):
            failures.append(f"{variant}: package {row.get('package')!r} does not match expected")
        if row.get("status") != "ok":
            failures.append(f"{variant}: status is {row.get('status')!r}, expected 'ok'")
        if row.get("failures") != []:
            failures.append(f"{variant}: failures are not empty")
        if mode == "source":
            for key in (
                "schema_count",
                "impl_count",
                "python_ops_ref_count",
                "loaded_metal_file_count",
                "loaded_metal_kernel_count",
                "host_kernel_ref_count",
                "host_kernel_field_count",
                "metal_kernel_count",
            ):
                if not _nonempty_positive_int(row.get(key)):
                    failures.append(f"{variant}: {key} must be positive")
            if row.get("impl_target_count") != row.get("impl_count"):
                failures.append(f"{variant}: impl_target_count does not match impl_count")
            if row.get("initialized_kernel_field_count") != row.get("host_kernel_field_count"):
                failures.append(f"{variant}: initialized_kernel_field_count does not match host_kernel_field_count")
        elif mode == "import":
            if not _nonempty_positive_int(row.get("schema_count")):
                failures.append(f"{variant}: schema_count must be positive")
            if row.get("registered_schema_count") != row.get("schema_count"):
                failures.append(f"{variant}: registered_schema_count does not match schema_count")
            if row.get("missing_registered_schemas") != []:
                failures.append(f"{variant}: missing_registered_schemas is not empty")
            extension_library = row.get("extension_library")
            if not isinstance(extension_library, str) or not extension_library.endswith(".so"):
                failures.append(f"{variant}: extension_library is not a built .so")
            elif not Path(extension_library).exists():
                failures.append(f"{variant}: extension_library does not exist")
            if row.get("import_error") != "":
                failures.append(f"{variant}: import_error is not empty")
            if row.get("extension_load_error") != "":
                failures.append(f"{variant}: extension_load_error is not empty")
            if not _nonempty_positive_int(row.get("compiled_source_count")):
                failures.append(f"{variant}: compiled_source_count must be positive")
    missing = sorted(set(expected) - seen)
    extra = sorted(seen - set(expected))
    if missing:
        failures.append(f"missing expected variants: {missing}")
    if extra:
        failures.append(f"unexpected variants: {extra}")
    return failures


def _smoke_bundle_failures(payload: dict[str, Any]) -> list[str]:
    failures = []
    if payload.get("quality_claim") is not False:
        failures.append("quality_claim must be false")
    if payload.get("speed_claim") is not False:
        failures.append("speed_claim must be false")
    if payload.get("scope") != "rebuilt_native_variant_smoke_artifacts_only":
        failures.append("scope is not rebuilt_native_variant_smoke_artifacts_only")
    required = payload.get("required")
    if payload.get("required_count") != len(REQUIRED_ARTIFACTS):
        failures.append(f"required_count is {payload.get('required_count')!r}, expected {len(REQUIRED_ARTIFACTS)}")
    if not isinstance(required, list):
        failures.append("required is missing or not a list")
        return failures
    expected = {str(spec["label"]): str(spec["benchmark"]) for spec in REQUIRED_ARTIFACTS}
    seen: set[str] = set()
    for row in required:
        if not isinstance(row, dict):
            failures.append("required smoke row is not an object")
            continue
        label = row.get("label")
        if not isinstance(label, str):
            failures.append("required smoke row has invalid label")
            continue
        seen.add(label)
        if row.get("status") != "ok":
            failures.append(f"{label}: status is {row.get('status')!r}, expected 'ok'")
        if row.get("artifact_status") != "ok":
            failures.append(f"{label}: artifact_status is {row.get('artifact_status')!r}, expected 'ok'")
        if not _path_exists(row.get("path")):
            failures.append(f"{label}: artifact path is missing or does not exist")
        if row.get("benchmark") != expected.get(label):
            failures.append(f"{label}: benchmark {row.get('benchmark')!r} does not match expected")
        if row.get("failures") != []:
            failures.append(f"{label}: failures are not empty")
    missing = sorted(set(expected) - seen)
    extra = sorted(seen - set(expected))
    if missing:
        failures.append(f"missing required smoke labels: {missing}")
    if extra:
        failures.append(f"unexpected smoke labels: {extra}")
    invalid = payload.get("known_invalid_tiled_ownerupdate")
    if not isinstance(invalid, dict):
        failures.append("known_invalid_tiled_ownerupdate is missing")
    else:
        if invalid.get("status") != "ok":
            failures.append("known_invalid_tiled_ownerupdate status is not ok")
        if invalid.get("present") is True and not _path_exists(invalid.get("path")):
            failures.append("known_invalid_tiled_ownerupdate path is missing or does not exist")
        if invalid.get("classification") != "expected_invalid_tiled_ownerupdate":
            failures.append("known_invalid_tiled_ownerupdate classification is not expected_invalid_tiled_ownerupdate")
        if invalid.get("failures") != []:
            failures.append("known_invalid_tiled_ownerupdate failures are not empty")
    return failures


def _artifact_status(
    path: Path,
    *,
    required_status: str = "ok",
    validator: str | None = None,
) -> dict[str, Any]:
    payload, error = _load_json(path)
    if payload is None:
        return {
            "path": str(path),
            "status": "missing_or_unreadable",
            "complete": False,
            "failures": [f"could not load artifact: {error}"],
        }
    status = payload.get("status")
    failures = [] if status == required_status else [f"status is {status!r}, expected {required_status!r}"]
    artifact_failures = payload.get("failures")
    if artifact_failures != []:
        failures.append("artifact failures are not empty")
    if validator == "source_wiring":
        failures.extend(_variant_bundle_failures(payload, mode="source"))
    elif validator == "import_registration":
        failures.extend(_variant_bundle_failures(payload, mode="import"))
    elif validator == "smoke_bundle":
        failures.extend(_smoke_bundle_failures(payload))
    return {
        "path": str(path),
        "status": status,
        "complete": not failures,
        "failures": failures,
        "payload": payload,
    }


def _next_mps_verifier_report(path: Path) -> dict[str, Any]:
    try:
        return verify_next_mps_summary(path)
    except Exception as exc:  # pragma: no cover - defensive audit path
        return {"status": "error", "failures": [f"next-MPS result verifier raised: {exc}"]}


def _path_value_matches(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return Path(value).resolve(strict=False) == expected.resolve(strict=False)


def _embedded_result_verifier_failures(payload: dict[str, Any], path: Path) -> list[str]:
    failures = []
    if payload.get("verify_result") is not True:
        failures.append("next-MPS summary was not launched with verify_result=true")
    if payload.get("result_verifier_returncode") != 0:
        failures.append("next-MPS embedded result_verifier_returncode is not 0")
    verifier_payload = payload.get("result_verifier_payload")
    if not isinstance(verifier_payload, dict):
        failures.append("next-MPS embedded result_verifier_payload is missing")
    elif verifier_payload.get("status") != "ok":
        failures.append("next-MPS embedded result_verifier_payload status is not ok")
    elif not _path_value_matches(verifier_payload.get("summary_path"), path):
        failures.append("next-MPS embedded result_verifier_payload summary_path does not target this summary")
    elif verifier_payload.get("artifact_checks_skipped") is not False:
        failures.append("next-MPS embedded result_verifier_payload skipped artifact checks")
    elif verifier_payload.get("failures") != []:
        failures.append("next-MPS embedded result_verifier_payload failures are not empty")
    else:
        artifact_value = payload.get("planned_worldfoam_artifact")
        if isinstance(artifact_value, str) and artifact_value:
            artifact_path = Path(artifact_value)
            if not _path_value_matches(verifier_payload.get("worldfoam_artifact"), artifact_path):
                failures.append("next-MPS embedded result_verifier_payload worldfoam_artifact does not match plan")
    command = payload.get("result_verifier_command")
    if not isinstance(command, list):
        failures.append("next-MPS result_verifier_command is missing")
    else:
        command_text = [str(item) for item in command]
        if not any(item.endswith("verify_worldfoam_next_mps_candidate_result.py") for item in command_text):
            failures.append("next-MPS result_verifier_command does not call the WorldFoam result verifier")
        if not any(Path(item).resolve(strict=False) == path.resolve(strict=False) for item in command_text):
            failures.append("next-MPS result_verifier_command does not target this summary")
    return failures


def default_next_mps_summary_json(results_dir: Path = RESULTS_DIR) -> Path:
    candidates = [path for path in results_dir.glob(NEXT_MPS_SUMMARY_GLOB) if path.is_file()]
    if not candidates:
        return results_dir / DEFAULT_NEXT_MPS_SUMMARY_JSON.name
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _brief_blocking_process(process: Any) -> dict[str, Any]:
    if not isinstance(process, dict):
        return {"raw": str(process)[:1024]}
    keys = {
        "pid",
        "ppid",
        "stat",
        "elapsed",
        "block_reason",
        "pcpu",
        "pmem",
        "command",
        "declared_duration_hours",
        "elapsed_s",
        "estimated_remaining_s",
        "estimated_done_at",
        "active_cpu",
        "summary_cpu_active",
        "live_cpu_active",
        "pid_live",
        "live_pcpu",
        "recent_output_count",
    }
    return {
        key: (str(value)[:1024] if key == "command" else value)
        for key, value in process.items()
        if key in keys
    }


def _current_benchmark_environment_report(probe: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(probe, dict):
        return None
    blocking_processes = probe.get("blocking_processes")
    if not isinstance(blocking_processes, list):
        blocking_processes = []
    blocker_summary = _external_blocker_summary(blocking_processes)
    blocking_kind_counts = blocker_summary.get("blocking_kind_counts")
    blocking_reason_counts = blocker_summary.get("blocking_reason_counts")
    blocking_screen_session_names = blocker_summary.get("blocking_screen_session_names")
    manual_next_actions = blocker_summary.get("manual_next_actions")
    report = dict(probe)
    report["blocking_kind_counts"] = blocking_kind_counts if isinstance(blocking_kind_counts, dict) else {}
    report["blocking_reason_counts"] = blocking_reason_counts if isinstance(blocking_reason_counts, dict) else {}
    report["blocking_screen_session_names"] = (
        [str(name) for name in blocking_screen_session_names]
        if isinstance(blocking_screen_session_names, list)
        else []
    )
    report["manual_next_actions"] = (
        [str(action) for action in manual_next_actions] if isinstance(manual_next_actions, list) else []
    )
    report["blocking_process_sample"] = [_brief_blocking_process(process) for process in blocking_processes]
    return report


def _diagnosis_matches_summary(summary_value: Any, next_mps_summary_json: Path) -> bool:
    if not isinstance(summary_value, str) or not summary_value:
        return False
    raw_path = Path(summary_value)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(ROOT / raw_path)
        candidates.append(next_mps_summary_json.parent / raw_path)
    expected = next_mps_summary_json.resolve(strict=False)
    return any(candidate.resolve(strict=False) == expected for candidate in candidates)


def _parse_checked_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _diagnosis_age_s(checked_at: Any, *, now: datetime | None) -> float | None:
    checked = _parse_checked_at(checked_at)
    if checked is None:
        return None
    if now is None:
        now = datetime.now(checked.tzinfo) if checked.tzinfo is not None else datetime.now()
    elif checked.tzinfo is not None and now.tzinfo is not None:
        now = now.astimezone(checked.tzinfo)
    elif checked.tzinfo is not None and now.tzinfo is None:
        now = now.replace(tzinfo=checked.tzinfo)
    elif checked.tzinfo is None and now.tzinfo is not None:
        checked = checked.replace(tzinfo=now.tzinfo)
    return max(0.0, (now - checked).total_seconds())


def _live_blocker_diagnosis_status(
    path: Path | None,
    *,
    next_mps_summary_json: Path,
    max_age_s: float | None = DEFAULT_MAX_BLOCKER_DIAGNOSIS_AGE_S,
    now: datetime | None = None,
) -> dict[str, Any]:
    if path is None:
        return {"available": False}
    payload, error = _load_json(path)
    if payload is None:
        return {
            "path": str(path),
            "available": False,
            "matches_next_mps_summary": False,
            "failures": [f"could not load blocker diagnosis: {error}"],
        }
    blockers = payload.get("blockers")
    checked_at = payload.get("checked_at")
    age_s = _diagnosis_age_s(checked_at, now=now)
    fresh = None if age_s is None or max_age_s is None else age_s <= max_age_s
    failures = []
    if fresh is False:
        failures.append(f"blocker diagnosis is stale: age_s={age_s:.1f} > max_age_s={max_age_s:.1f}")
    blocker_summary = _external_blocker_summary(blockers if isinstance(blockers, list) else [])
    blocking_screen_session_names = blocker_summary.get("blocking_screen_session_names")
    return {
        "path": str(path),
        "available": True,
        "matches_next_mps_summary": _diagnosis_matches_summary(
            payload.get("summary_json"), next_mps_summary_json
        ),
        "summary_json": payload.get("summary_json"),
        "checked_at": checked_at,
        "diagnosis_age_s": age_s,
        "diagnosis_max_age_s": max_age_s,
        "diagnosis_fresh": fresh,
        "failures": failures,
        "status": payload.get("status"),
        "process_sample_limit": payload.get("process_sample_limit"),
        "blocker_count": payload.get("blocker_count"),
        "blocker_sample_count": payload.get("blocker_sample_count"),
        "blocker_unlisted_count": payload.get("blocker_unlisted_count"),
        "contending_process_count": payload.get("contending_process_count"),
        "contending_process_sample_count": payload.get("contending_process_sample_count"),
        "contending_process_unlisted_count": payload.get("contending_process_unlisted_count"),
        "live_blocker_count": payload.get("live_blocker_count"),
        "recent_output_blocker_count": payload.get("recent_output_blocker_count"),
        "live_or_recent_blocker_count": payload.get("live_or_recent_blocker_count"),
        "category_counts": payload.get("category_counts") if isinstance(payload.get("category_counts"), dict) else {},
        "live_category_counts": payload.get("live_category_counts")
        if isinstance(payload.get("live_category_counts"), dict)
        else {},
        "active_cpu_category_counts": payload.get("active_cpu_category_counts")
        if isinstance(payload.get("active_cpu_category_counts"), dict)
        else {},
        "summary_cpu_active_category_counts": payload.get("summary_cpu_active_category_counts")
        if isinstance(payload.get("summary_cpu_active_category_counts"), dict)
        else {},
        "live_cpu_over_preflight_threshold_category_counts": payload.get(
            "live_cpu_over_preflight_threshold_category_counts"
        )
        if isinstance(payload.get("live_cpu_over_preflight_threshold_category_counts"), dict)
        else {},
        "recent_output_category_counts": payload.get("recent_output_category_counts")
        if isinstance(payload.get("recent_output_category_counts"), dict)
        else {},
        "max_estimated_remaining_s_by_category": payload.get("max_estimated_remaining_s_by_category")
        if isinstance(payload.get("max_estimated_remaining_s_by_category"), dict)
        else {},
        "blocking_screen_session_names": [str(name) for name in blocking_screen_session_names]
        if isinstance(blocking_screen_session_names, list)
        else [],
        "blockers": [_brief_blocking_process(process) for process in blockers] if isinstance(blockers, list) else [],
    }


def _latest_estimated_done_at(live_blocker_diagnosis: dict[str, Any]) -> str | None:
    best_text = None
    best_dt = None
    blockers = live_blocker_diagnosis.get("blockers")
    if not isinstance(blockers, list):
        return None
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        text = blocker.get("estimated_done_at")
        parsed = _parse_checked_at(text)
        if parsed is None:
            continue
        if best_dt is None or parsed > best_dt:
            best_text = str(text)
            best_dt = parsed
    return best_text


def _clean_mps_rerun_plan(
    *,
    next_mps: dict[str, Any],
    live_blocker_diagnosis: dict[str, Any],
    current_benchmark_environment_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_probe = _current_benchmark_environment_report(current_benchmark_environment_probe)
    available = live_blocker_diagnosis.get("available") is True
    live_status = live_blocker_diagnosis.get("status") if available else None
    live_blocked = live_status == "blocked"
    fresh = live_blocker_diagnosis.get("diagnosis_fresh")
    known_stale = fresh is False
    current_probe_present = current_probe is not None
    current_probe_available = (
        current_probe.get("available") is True if current_probe_present else None
    )
    current_probe_blocks = (
        current_probe.get("blocks_promotion") is True if current_probe_present else False
    )
    current_probe_failed = current_probe_present and current_probe_available is not True
    ready_now = available and not live_blocked and not known_stale and not current_probe_failed and not current_probe_blocks
    blocking_conditions = []
    if not available:
        blocking_conditions.append("live_blocker_diagnosis_unavailable")
    if live_blocked:
        blocking_conditions.append("live_or_recent_external_blockers_present")
    if known_stale:
        blocking_conditions.append("live_blocker_diagnosis_stale")
    if current_probe_failed:
        blocking_conditions.append("current_benchmark_environment_probe_failed")
    if current_probe_blocks:
        blocking_conditions.append("current_benchmark_environment_contended")
    if ready_now:
        wait_reason = None
    elif not available or live_blocked or known_stale:
        wait_reason = "live blocker diagnosis must be fresh and report no active/recent external contenders"
    elif current_probe_failed:
        wait_reason = "current benchmark environment probe must succeed before a clean MPS gate run"
    else:
        wait_reason = "current benchmark environment probe must report ok/background before a clean MPS gate run"
    estimated_done_at = _latest_estimated_done_at(live_blocker_diagnosis) if available else None
    current_blocking_kind_counts = current_probe.get("blocking_kind_counts") if current_probe_present else {}
    if not isinstance(current_blocking_kind_counts, dict):
        current_blocking_kind_counts = {}
    return {
        "cwd": str(ROOT),
        "command": list(CLEAN_MPS_RERUN_COMMAND),
        "requires_quiet_window": True,
        "ready_to_run_now": ready_now,
        "wait_reason": wait_reason,
        "blocking_conditions": blocking_conditions,
        "latest_blocker_checked_at": live_blocker_diagnosis.get("checked_at") if available else None,
        "run_after_estimated_done_at": estimated_done_at,
        "run_after_estimated_done_at_scope": "live_blocker_diagnosis_only" if estimated_done_at else None,
        "run_after_estimated_done_at_requires_reprobe": bool(estimated_done_at or current_probe_present),
        "live_max_estimated_remaining_s_by_category": live_blocker_diagnosis.get(
            "max_estimated_remaining_s_by_category"
        )
        if available and isinstance(live_blocker_diagnosis.get("max_estimated_remaining_s_by_category"), dict)
        else {},
        "current_benchmark_environment_has_independent_blockers": bool(current_blocking_kind_counts),
        "live_blocker_status": live_status,
        "live_blocker_count": live_blocker_diagnosis.get("live_blocker_count") if available else None,
        "live_blocking_category_counts": live_blocker_diagnosis.get("live_category_counts") if available else {},
        "live_blocking_screen_session_names": live_blocker_diagnosis.get("blocking_screen_session_names")
        if available and isinstance(live_blocker_diagnosis.get("blocking_screen_session_names"), list)
        else [],
        "preflight_sample_category_counts": live_blocker_diagnosis.get("category_counts") if available else {},
        "live_recent_output_category_counts": live_blocker_diagnosis.get("recent_output_category_counts")
        if available
        else {},
        "current_benchmark_environment_probe_available": current_probe_available,
        "current_benchmark_environment_status": current_probe.get("status") if current_probe_present else None,
        "current_benchmark_environment_blocks_promotion": current_probe_blocks if current_probe_present else None,
        "current_benchmark_environment_returncode": current_probe.get("returncode") if current_probe_present else None,
        "current_benchmark_environment_blocking_process_count": current_probe.get("blocking_process_count")
        if current_probe_present
        else None,
        "current_benchmark_environment_blocking_kind_counts": current_blocking_kind_counts,
        "current_benchmark_environment_blocking_reason_counts": current_probe.get("blocking_reason_counts")
        if current_probe_present
        else {},
        "current_benchmark_environment_blocking_screen_session_names": current_probe.get(
            "blocking_screen_session_names"
        )
        if current_probe_present
        else [],
        "current_benchmark_environment_manual_next_actions": current_probe.get("manual_next_actions")
        if current_probe_present
        else [],
        "current_benchmark_environment_blocking_process_sample": current_probe.get("blocking_process_sample")
        if current_probe_present
        else [],
        "latest_preflight_status": next_mps.get("status"),
        "acceptance_gate": "do not promote until verify_worldfoam_next_mps_candidate_result reports ok on a clean real32 MPS run",
        "embedded_result_verification": "--verify-result" in CLEAN_MPS_RERUN_COMMAND,
        "acceptance_verifier_required_status": "ok",
        "acceptance_verifier_command_template": list(CLEAN_MPS_RESULT_VERIFIER_COMMAND_TEMPLATE),
    }


def _next_mps_status(path: Path) -> dict[str, Any]:
    payload, error = _load_json(path)
    if payload is None:
        return {
            "path": str(path),
            "status": "missing_or_unreadable",
            "complete": False,
            "blocked": False,
            "failures": [f"could not load next-MPS summary: {error}"],
        }
    status = payload.get("status")
    verifier_report = _next_mps_verifier_report(path)
    verifier_status = verifier_report.get("status")
    verifier_failures = verifier_report.get("failures")
    if not isinstance(verifier_failures, list):
        verifier_failures = []
    embedded_verifier_failures = _embedded_result_verifier_failures(payload, path)
    embedded_verifier_ok = not embedded_verifier_failures
    complete = verifier_status == "ok" and embedded_verifier_ok
    blocked = status in BLOCKED_NEXT_MPS_STATUSES
    if complete:
        failures = []
    elif blocked:
        failures = []
    else:
        failures = [f"next-MPS summary status is {status!r}"]
        failures.extend(str(failure) for failure in verifier_failures)
        failures.extend(embedded_verifier_failures)
    blocking_processes = payload.get("preflight_blocking_processes")
    if isinstance(blocking_processes, list):
        blockers = _external_blocker_summary(blocking_processes)
    else:
        blockers = payload.get("preflight_external_blocker_summary")
    blocking_counts = blockers.get("blocking_kind_counts") if isinstance(blockers, dict) else {}
    blocking_reason_counts = blockers.get("blocking_reason_counts") if isinstance(blockers, dict) else {}
    blocking_screen_session_names = blockers.get("blocking_screen_session_names") if isinstance(blockers, dict) else []
    manual_next_actions = blockers.get("manual_next_actions") if isinstance(blockers, dict) else []
    return {
        "path": str(path),
        "status": status,
        "history_jsonl": payload.get("history_jsonl"),
        "complete": complete,
        "blocked": blocked,
        "failures": failures,
        "result_verifier_status": verifier_status,
        "result_verifier_failures": [str(failure) for failure in verifier_failures],
        "embedded_result_verification_complete": embedded_verifier_ok,
        "embedded_result_verifier_failures": embedded_verifier_failures,
        "result_verifier_worldfoam_artifact": verifier_report.get("worldfoam_artifact"),
        "result_verifier_artifact_checks_skipped": verifier_report.get("artifact_checks_skipped"),
        "blocking_kind_counts": blocking_counts if isinstance(blocking_counts, dict) else {},
        "blocking_reason_counts": blocking_reason_counts if isinstance(blocking_reason_counts, dict) else {},
        "blocking_screen_session_names": [str(name) for name in blocking_screen_session_names]
        if isinstance(blocking_screen_session_names, list)
        else [],
        "manual_next_actions": [str(action) for action in manual_next_actions]
        if isinstance(manual_next_actions, list)
        else [],
        "preflight_blocking_processes": [_brief_blocking_process(process) for process in blocking_processes]
        if isinstance(blocking_processes, list)
        else [],
        "preflight_process_sample_limit": payload.get("preflight_process_sample_limit"),
        "preflight_blocking_process_count": payload.get("preflight_blocking_process_count"),
        "preflight_blocking_process_sample_count": payload.get("preflight_blocking_process_sample_count"),
        "preflight_blocking_process_unlisted_count": payload.get("preflight_blocking_process_unlisted_count"),
        "preflight_contending_process_count": payload.get("preflight_contending_process_count"),
        "preflight_contending_process_sample_count": payload.get("preflight_contending_process_sample_count"),
        "preflight_contending_process_unlisted_count": payload.get("preflight_contending_process_unlisted_count"),
        "preflight_blocking_reasons": payload.get("preflight_blocking_reasons"),
        "preflight_attempt_count": payload.get("preflight_attempt_count"),
        "preflight_retry_timeout_s": payload.get("preflight_retry_timeout_s"),
        "preflight_stability_samples_requested": payload.get("preflight_stability_samples_requested"),
        "preflight_stability_samples_completed": payload.get("preflight_stability_samples_completed"),
        "preflight_stability_ok": payload.get("preflight_stability_ok"),
        "planned_worldfoam_artifact": payload.get("planned_worldfoam_artifact"),
    }


def audit(
    *,
    source_json: Path = DEFAULT_SOURCE_JSON,
    import_json: Path = DEFAULT_IMPORT_JSON,
    smoke_bundle_json: Path = DEFAULT_SMOKE_BUNDLE_JSON,
    next_mps_summary_json: Path | None = None,
    blocker_diagnosis_json: Path | None = None,
    current_benchmark_environment_probe: dict[str, Any] | None = None,
    max_blocker_diagnosis_age_s: float | None = DEFAULT_MAX_BLOCKER_DIAGNOSIS_AGE_S,
    now: datetime | None = None,
) -> dict[str, Any]:
    use_default_blocker_diagnosis = blocker_diagnosis_json is None and next_mps_summary_json is None
    if next_mps_summary_json is None:
        next_mps_summary_json = default_next_mps_summary_json()
    if use_default_blocker_diagnosis:
        blocker_diagnosis_json = DEFAULT_BLOCKER_DIAGNOSIS_JSON
    source = _artifact_status(source_json, validator="source_wiring")
    imports = _artifact_status(import_json, validator="import_registration")
    smoke_bundle = _artifact_status(smoke_bundle_json, validator="smoke_bundle")
    next_mps = _next_mps_status(next_mps_summary_json)
    live_blocker_diagnosis = _live_blocker_diagnosis_status(
        blocker_diagnosis_json,
        next_mps_summary_json=next_mps_summary_json,
        max_age_s=max_blocker_diagnosis_age_s,
        now=now,
    )
    current_benchmark_environment = _current_benchmark_environment_report(current_benchmark_environment_probe)
    fixed_requirements = {
        "native_source_wiring": source["complete"],
        "native_import_registration": imports["complete"],
        "rebuilt_native_smoke_bundle": smoke_bundle["complete"],
    }
    missing_requirements = {
        "clean_real32_mps_psnr_speed_sublinear_gate": not next_mps["complete"],
    }
    failures = []
    for label, item in (("source", source), ("import", imports), ("smoke_bundle", smoke_bundle), ("next_mps", next_mps)):
        failures.extend(f"{label}: {failure}" for failure in item.get("failures", []))

    shader_forks_fixed = all(fixed_requirements.values())
    objective_complete = shader_forks_fixed and next_mps["complete"]
    live_diagnosis_available = live_blocker_diagnosis.get("available") is True
    live_diagnosis_fresh = live_blocker_diagnosis.get("diagnosis_fresh") is not False
    live_gate_blocked = live_blocker_diagnosis.get("status") == "blocked"
    current_probe_present = current_benchmark_environment is not None
    current_probe_failed = (
        current_benchmark_environment.get("available") is not True if current_probe_present else False
    )
    current_probe_blocked = (
        current_benchmark_environment.get("blocks_promotion") is True if current_probe_present else False
    )
    live_gate_clear = (
        live_diagnosis_available
        and live_diagnosis_fresh
        and not live_gate_blocked
        and not current_probe_failed
        and not current_probe_blocked
    )
    if objective_complete:
        status = "complete"
    elif shader_forks_fixed and next_mps["blocked"] and live_gate_clear:
        status = "incomplete_ready_for_clean_mps_gate"
    elif shader_forks_fixed and next_mps["blocked"]:
        status = "blocked_external_environment"
    elif shader_forks_fixed:
        status = "incomplete_missing_clean_mps_gate"
    else:
        status = "failed_prerequisite"
    clean_mps_rerun_plan = _clean_mps_rerun_plan(
        next_mps=next_mps,
        live_blocker_diagnosis=live_blocker_diagnosis,
        current_benchmark_environment_probe=current_benchmark_environment,
    )

    return {
        "status": status,
        "objective_complete": objective_complete,
        "shader_fork_smoke_state_fixed": shader_forks_fixed,
        "quality_claim": False,
        "speed_claim": False,
        "fixed_requirements": fixed_requirements,
        "missing_requirements": missing_requirements,
        "artifacts": {
            "source_wiring": {key: value for key, value in source.items() if key != "payload"},
            "import_registration": {key: value for key, value in imports.items() if key != "payload"},
            "rebuilt_native_smoke_bundle": {key: value for key, value in smoke_bundle.items() if key != "payload"},
            "next_mps_quality_speed": next_mps,
            "live_blocker_diagnosis": live_blocker_diagnosis,
            "current_benchmark_environment_probe": current_benchmark_environment,
        },
        "clean_mps_rerun_plan": clean_mps_rerun_plan,
        "failures": failures,
        "next_actions": [
            "rerun next-MPS candidate only after preflight reports a quiet external-process window",
            "use clean_mps_rerun_plan.command for the guarded real32 MPS PSNR/speed/sublinear gate",
            "do not promote WorldFoam real32 PSNR/speed/sublinear claims from source/import/smoke evidence alone",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report WorldFoam fork-shader goal state without promoting weak evidence.")
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--import-json", type=Path, default=DEFAULT_IMPORT_JSON)
    parser.add_argument("--smoke-bundle-json", type=Path, default=DEFAULT_SMOKE_BUNDLE_JSON)
    parser.add_argument("--next-mps-summary-json", type=Path)
    parser.add_argument("--blocker-diagnosis-json", type=Path)
    parser.add_argument("--max-blocker-diagnosis-age-s", type=float, default=DEFAULT_MAX_BLOCKER_DIAGNOSIS_AGE_S)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = audit(
        source_json=args.source_json,
        import_json=args.import_json,
        smoke_bundle_json=args.smoke_bundle_json,
        next_mps_summary_json=args.next_mps_summary_json,
        blocker_diagnosis_json=args.blocker_diagnosis_json,
        max_blocker_diagnosis_age_s=args.max_blocker_diagnosis_age_s,
    )
    if args.out_json is not None:
        _write_json_atomic(args.out_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] != "failed_prerequisite" else 1


if __name__ == "__main__":
    raise SystemExit(main())
