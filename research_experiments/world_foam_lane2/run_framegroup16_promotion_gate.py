#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DYNAWORLD = Path(__file__).resolve().parents[2]
LANE_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2"
RESULTS_DIR = LANE_DIR / "results"
VARIANT_ROOT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_fused_slab_v0"
TRAIN_SRC = DYNAWORLD / "src" / "train"
TRAIN_EVAL = LANE_DIR / "train_eval_owner_run_tape.py"
VERIFY = LANE_DIR / "verify_framegroup16_timing_robust.py"
NATIVE_PACKED_EXTENSION_VERIFY = LANE_DIR / "verify_native_packed_extension.py"
DEFAULT_REFERENCE = (
    RESULTS_DIR / "2026-05-18_gate4_endpoint_record_nativechunk_directdelta_repeat20_render64_site24_2_4_8_16.json"
)
DEFAULT_FRAME_COUNTS = "2,4,8,16"
DEFAULT_FRAME_COUNT_TUPLE = (2, 4, 8, 16)
DEFAULT_TAPE_MODE = "endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse"
DEFAULT_STABLE_PREFLIGHT_CHECKS = 2
NATIVE_VARIANT_FLAGS = (
    ("experimental_native_cut_prep_delta", "--experimental-native-cut-prep-delta"),
    ("experimental_native_sorted_delta", "--experimental-native-sorted-delta"),
    ("experimental_native_pack_records", "--experimental-native-pack-records"),
    ("experimental_native_emitted_pack_records", "--experimental-native-emitted-pack-records"),
)
TRAIN_VARIANT_FLAGS = (
    *NATIVE_VARIANT_FLAGS,
    ("experimental_minimal_packed_delta_device", "--experimental-minimal-packed-delta-device"),
    ("experimental_kernel_order_packed_delta_device", "--experimental-kernel-order-packed-delta-device"),
    ("experimental_smallrun16_packed_delta", "--experimental-smallrun16-packed-delta"),
    ("experimental_launch_only_packed_delta", "--experimental-launch-only-packed-delta"),
    ("experimental_unchecked_launch_only_packed_delta", "--experimental-unchecked-launch-only-packed-delta"),
    ("experimental_reduce32_launch_only_packed_delta", "--experimental-reduce32-launch-only-packed-delta"),
    ("experimental_rowselect32_launch_only_packed_delta", "--experimental-rowselect32-launch-only-packed-delta"),
    ("experimental_rowdesc_launch_only_packed_delta", "--experimental-rowdesc-launch-only-packed-delta"),
    ("experimental_rowdesc32_launch_only_packed_delta", "--experimental-rowdesc32-launch-only-packed-delta"),
    ("experimental_cpu_rebase_delta", "--experimental-cpu-rebase-delta"),
)
REQUIRED_NATIVE_PACKED_VERIFY_VALUES = {
    "variant_root": str(VARIANT_ROOT),
    "base_offsets_i32": [0, 2],
    "base_offsets_i32_dtype": "int32",
    "base_record_i32": [2097152, 1049089],
    "base_record_i32_dtype": "int32",
    "change_record_i32": [],
    "change_record_i32_dtype": "int32",
    "track_change_offsets_i32": [0, 0],
    "track_change_offsets_i32_dtype": "int32",
    "cut_base_offsets_i32": [0, 2],
    "cut_base_offsets_i32_dtype": "int32",
    "cut_base_record_i32": [2097152, 1049089],
    "cut_base_record_i32_dtype": "int32",
    "cut_change_record_i32": [],
    "cut_change_record_i32_dtype": "int32",
    "cut_track_change_offsets_i32": [0, 0],
    "cut_track_change_offsets_i32_dtype": "int32",
    "cut_array_cut_ids_i64": [-1, 0, -2, -1, 0, -2],
    "cut_array_cut_offsets_i64": [0, 3, 6],
    "cut_array_start_segments_i64": [0, 0],
    "cut_array_initial_owner_i64": [0, 0],
    "changing_sorted_change_frame_i32": [1],
    "changing_sorted_change_offsets_i32": [0, 2],
    "changing_sorted_track_change_offsets_i32": [0, 1],
    "changing_sorted_change_record_i32": [2097153, 1049088],
    "changing_sorted_change_record_i32_dtype": "int32",
    "changing_cut_change_frame_i32": [1],
    "changing_cut_change_offsets_i32": [0, 2],
    "changing_cut_track_change_offsets_i32": [0, 1],
    "changing_cut_change_record_i32": [2097153, 1049088],
    "changing_cut_change_record_i32_dtype": "int32",
    "has_launch_only_packed_framegroup16_op": True,
    "has_launch_only_packed_framegroup16_unchecked_op": True,
    "has_launch_only_packed_framegroup16_reduce32_op": True,
    "has_launch_only_packed_framegroup16_reduce32_unchecked_op": True,
    "has_launch_only_packed_framegroup16_rowselect32_op": True,
    "has_launch_only_packed_framegroup16_rowselect32_unchecked_op": True,
    "has_launch_only_packed_framegroup16_rowdesc_op": True,
    "has_launch_only_packed_framegroup16_rowdesc_unchecked_op": True,
    "has_launch_only_packed_framegroup16_rowdesc32_op": True,
    "has_launch_only_packed_framegroup16_rowdesc32_unchecked_op": True,
    "has_launch_only_packed_framegroup16_recompute_op": True,
    "has_launch_only_packed_framegroup16_smallrun16_op": True,
    "has_launch_only_packed_framegroup16_materialized_op": True,
}
for _native_verify_guard_key in (
    "pack_endpoint_records_i32_rejects_rank2",
    "pack_endpoint_records_i32_rejects_owner_out_of_range",
    "pack_endpoint_records_i32_rejects_cut_out_of_range",
):
    REQUIRED_NATIVE_PACKED_VERIFY_VALUES[_native_verify_guard_key] = True
for _native_verify_prefix in (
    "gate4_delta_replace_from_cuts",
    "gate4_delta_replace_packed_from_cuts",
):
    for _native_verify_suffix in (
        "start_segment_oob",
        "active_mismatch",
        "boundary_other_oob",
        "nan_depth",
        "decreasing_depth",
        "bad_first_sentinel",
        "bad_last_sentinel",
        "internal_boundary_id_oob",
        "single_cut_row",
    ):
        REQUIRED_NATIVE_PACKED_VERIFY_VALUES[f"{_native_verify_prefix}_rejects_{_native_verify_suffix}"] = True
for _native_verify_prefix in (
    "gate4_delta_replace_from_sorted",
    "gate4_delta_replace_packed_from_sorted",
):
    for _native_verify_suffix in (
        "row_active_bad_value",
        "valid_count_oob",
        "negative_boundary_id",
        "boundary_id_oob",
        "boundary_other_oob",
        "nan_depth",
        "below_near_depth",
        "above_far_depth",
        "decreasing_depth",
    ):
        REQUIRED_NATIVE_PACKED_VERIFY_VALUES[f"{_native_verify_prefix}_rejects_{_native_verify_suffix}"] = True
for _native_verify_suffix in (
    "row_active_bad_value",
    "valid_count_oob",
    "negative_boundary_id",
    "nan_depth",
    "below_near_depth",
    "above_far_depth",
    "decreasing_depth",
):
    REQUIRED_NATIVE_PACKED_VERIFY_VALUES[f"gate4_cut_arrays_from_sorted_rejects_{_native_verify_suffix}"] = True
for _native_verify_key, _native_verify_value in tuple(REQUIRED_NATIVE_PACKED_VERIFY_VALUES.items()):
    if _native_verify_key.endswith(("_i32", "_i64")) and isinstance(_native_verify_value, list):
        _native_verify_dtype = "int32" if _native_verify_key.endswith("_i32") else "int64"
        REQUIRED_NATIVE_PACKED_VERIFY_VALUES.setdefault(f"{_native_verify_key}_dtype", _native_verify_dtype)
        REQUIRED_NATIVE_PACKED_VERIFY_VALUES[f"{_native_verify_key}_device"] = "cpu"
        REQUIRED_NATIVE_PACKED_VERIFY_VALUES[f"{_native_verify_key}_shape"] = [len(_native_verify_value)]
        REQUIRED_NATIVE_PACKED_VERIFY_VALUES[f"{_native_verify_key}_contiguous"] = True
del (
    _native_verify_dtype,
    _native_verify_guard_key,
    _native_verify_key,
    _native_verify_prefix,
    _native_verify_suffix,
    _native_verify_value,
)


def _repo_python() -> Path:
    candidate = DYNAWORLD / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(VARIANT_ROOT), str(LANE_DIR), str(TRAIN_SRC)]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _default_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_framegroup16_promotion_%H%M%S")


def _run(cmd: list[str], *, dry_run: bool) -> int:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=DYNAWORLD, env=_env(), check=False).returncode


def _run_json_command(cmd: list[str], *, dry_run: bool) -> tuple[int, dict[str, object] | None]:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0, None
    result = subprocess.run(cmd, cwd=DYNAWORLD, env=_env(), check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError:
        return result.returncode, None
    return result.returncode, decoded if isinstance(decoded, dict) else None


def _enabled_native_variant_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    for attr, flag in NATIVE_VARIANT_FLAGS:
        if bool(getattr(args, attr, False)):
            flags.append(flag)
    return flags


def _enabled_train_variant_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    for attr, flag in TRAIN_VARIANT_FLAGS:
        if bool(getattr(args, attr, False)):
            flags.append(flag)
    return flags


def _enabled_expected_payload_bools(args: argparse.Namespace) -> list[str]:
    expectations: list[str] = []
    for attr, _flag in TRAIN_VARIANT_FLAGS:
        if bool(getattr(args, attr, False)):
            expectations.extend(["--expect-payload-bool", f"{attr}=true"])
    return expectations


def _requires_native_packed_extension_verify(args: argparse.Namespace) -> bool:
    return bool(args.experimental_native_pack_records or args.experimental_native_emitted_pack_records)


def _native_packed_verify_failures(result: dict[str, object] | None, *, dry_run: bool) -> list[str]:
    if result is None:
        return [] if dry_run else ["native packed extension verifier did not return a JSON object"]
    failures: list[str] = []
    if result.get("status") != "ok":
        failures.append(f"native packed extension verifier status was {result.get('status')!r}, expected 'ok'")
    for key, expected in REQUIRED_NATIVE_PACKED_VERIFY_VALUES.items():
        actual = result.get(key)
        if actual != expected:
            failures.append(f"native packed extension verifier {key}={actual!r}, expected {expected!r}")
    return failures


def _preflight_progress_fields(
    attempts: list[dict[str, object]],
    *,
    required_successes: int,
) -> dict[str, object]:
    streaks = [
        int(attempt["success_streak"])
        for attempt in attempts
        if isinstance(attempt.get("success_streak"), int)
    ]
    return {
        "preflight_required_success_streak": max(1, int(required_successes)),
        "preflight_current_success_streak": streaks[-1] if streaks else 0,
        "preflight_max_success_streak": max(streaks) if streaks else 0,
    }


def _preflight_failure_reason(
    preflight_status: int,
    attempts: list[dict[str, object]],
    *,
    required_successes: int,
) -> str | None:
    if preflight_status == 0:
        return None
    progress = _preflight_progress_fields(attempts, required_successes=required_successes)
    if int(progress["preflight_max_success_streak"]) < max(1, int(required_successes)):
        if int(progress["preflight_max_success_streak"]) > 0:
            return "stable_preflight_streak_not_reached"
        return "benchmark_environment_never_clean"
    return "benchmark_environment_preflight_failed"


def _parse_frame_counts(value: str) -> tuple[int, ...]:
    frames = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(frames) < 1:
        raise ValueError("frame-counts must contain at least one integer")
    if tuple(sorted(frames)) != frames:
        raise ValueError("frame-counts must be sorted ascending")
    return frames


def _artifact_frame_counts(path: Path) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    rows = payload.get("rows")
    frames: list[int] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("frame_count"), int):
                frames.append(int(row["frame_count"]))
    elif isinstance(rows, dict):
        for key in rows:
            frames.append(int(key))
    else:
        raise ValueError(f"{path} rows must be a list or object keyed by frame count")
    return tuple(sorted(frames))


def _run_preflight(
    cmd: list[str],
    *,
    dry_run: bool,
    wait: bool,
    timeout_s: float,
    interval_s: float,
    stable_checks: int,
    summary: dict[str, object] | None = None,
    summary_path: Path | None = None,
) -> tuple[int, list[dict[str, object]]]:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0, []

    deadline = time.monotonic() + max(timeout_s, 0.0)
    required_successes = max(1, int(stable_checks))
    success_streak = 0
    attempts: list[dict[str, object]] = []
    while True:
        result = subprocess.run(cmd, cwd=DYNAWORLD, env=_env(), check=False, capture_output=True, text=True)
        if result.returncode == 0:
            success_streak += 1
        else:
            success_streak = 0
        snapshot: dict[str, object] | None = None
        try:
            decoded = json.loads(result.stdout)
            if isinstance(decoded, dict):
                snapshot = decoded
        except json.JSONDecodeError:
            snapshot = None
        attempt: dict[str, object] = {
            "returncode": result.returncode,
            "success_streak": success_streak,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if snapshot is not None:
            attempt["status"] = snapshot.get("status")
            attempt["blocking_processes"] = snapshot.get("blocking_processes", [])
            blocking = snapshot.get("blocking_processes")
            if isinstance(blocking, list) and blocking:
                top = blocking[0]
                if isinstance(top, dict):
                    attempt["top_blocking_process"] = {
                        "pid": top.get("pid"),
                        "pcpu": top.get("pcpu"),
                        "command": top.get("command"),
                    }
        elif result.stdout:
            attempt["stdout"] = result.stdout[-2000:]
        if result.stderr:
            attempt["stderr"] = result.stderr[-2000:]
        attempts.append(attempt)
        if summary is not None and summary_path is not None:
            summary["preflight_status"] = result.returncode
            summary["preflight_attempts"] = attempts
            summary.update(
                _preflight_progress_fields(
                    attempts,
                    required_successes=required_successes,
                )
            )
            if result.returncode == 0 and success_streak < required_successes:
                summary["status"] = "waiting_for_stable_preflight"
            elif wait and result.returncode != 0:
                summary["status"] = "waiting_for_preflight"
            else:
                summary["status"] = "preflight_checked"
            _write_summary(summary_path, summary)

        if result.returncode == 0 and success_streak >= required_successes:
            print(result.stdout, end="", flush=True)
            return 0, attempts
        if result.returncode == 0:
            print(result.stdout, end="", flush=True)
        if not wait or time.monotonic() >= deadline:
            if result.returncode != 0:
                print(result.stdout, end="", flush=True)
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr, flush=True)
            if result.returncode == 0 and success_streak < required_successes:
                return 2, attempts
            return result.returncode, attempts

        top_message = ""
        top = attempt.get("top_blocking_process")
        if isinstance(top, dict):
            top_message = f" top_blocking={top.get('pcpu')}% {top.get('command')}"
        print(
            f"[promotion_gate] preflight not stable; retrying in {interval_s:.1f}s."
            f" success_streak={success_streak}/{required_successes}.{top_message}",
            flush=True,
        )
        time.sleep(max(interval_s, 0.1))


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric_ms(row: dict[str, object], group: str) -> float | None:
    summary = row.get(group)
    if not isinstance(summary, dict):
        return None
    value = summary.get("median_ms")
    return float(value) if isinstance(value, (int, float)) else None


def _load_verify_brief(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    rows = payload.get("rows")
    row_briefs: dict[str, object] = {}
    if isinstance(rows, dict):
        for frame, row in rows.items():
            if not isinstance(row, dict):
                continue
            row_briefs[str(frame)] = {
                "total_median_ms": _metric_ms(row, "total"),
                "backward_median_ms": _metric_ms(row, "backward"),
                "storage_bytes": row.get("storage_bytes"),
                "topology_storage_bytes": row.get("topology_storage_bytes"),
                "coeff_storage_bytes": row.get("coeff_storage_bytes"),
                "mps_resident_storage_bytes": row.get("mps_resident_storage_bytes"),
                "mps_resident_noncoeff_storage_bytes": row.get("mps_resident_noncoeff_storage_bytes"),
                "mps_resident_coeff_storage_bytes": row.get("mps_resident_coeff_storage_bytes"),
                "heldout_psnr": row.get("heldout_psnr"),
            }
    return {
        "status": payload.get("status"),
        "clean_speedscale_artifact": payload.get("clean_speedscale_artifact"),
        "promoted_path_not_regressed": payload.get("promoted_path_not_regressed"),
        "expected_payload_bools": payload.get("expected_payload_bools", {}),
        "storage_scale": payload.get("storage_scale"),
        "topology_storage_scale": payload.get("topology_storage_scale"),
        "coeff_storage_scale": payload.get("coeff_storage_scale"),
        "mps_resident_storage_scale": payload.get("mps_resident_storage_scale"),
        "mps_resident_noncoeff_storage_scale": payload.get("mps_resident_noncoeff_storage_scale"),
        "mps_resident_coeff_storage_scale": payload.get("mps_resident_coeff_storage_scale"),
        "contamination": payload.get("contamination", []),
        "failures": payload.get("failures", []),
        "rows": row_briefs,
    }


def _attempt_path(path: Path, attempt_index: int, max_attempts: int) -> Path:
    if max_attempts <= 1:
        return path
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}.attempt{attempt_index}{suffix}")


def _attempt_artifact_paths(
    out_json: Path,
    partial_out_json: Path,
    verify_json: Path,
    *,
    max_attempts: int,
) -> list[tuple[Path, Path, Path]]:
    return [
        (
            _attempt_path(out_json, attempt_index, max_attempts),
            _attempt_path(partial_out_json, attempt_index, max_attempts),
            _attempt_path(verify_json, attempt_index, max_attempts),
        )
        for attempt_index in range(1, max_attempts + 1)
    ]


def _verify_brief_has_contamination(verify_brief: dict[str, object] | None) -> bool:
    if verify_brief is None:
        return False
    contamination = verify_brief.get("contamination")
    return isinstance(contamination, list) and bool(contamination)


def _verify_failure_is_structural(failure: object) -> bool:
    if not isinstance(failure, str):
        return False
    lower = failure.lower()
    structural_markers = (
        "storage scale",
        "topology storage scale",
        "mps resident",
        "coeff storage scale",
        "selected tape storage",
    )
    return any(marker in lower for marker in structural_markers)


def _verify_brief_is_retryable_contamination(verify_brief: dict[str, object] | None) -> bool:
    if not _verify_brief_has_contamination(verify_brief):
        return False
    failures = verify_brief.get("failures") if verify_brief is not None else None
    if isinstance(failures, list) and any(_verify_failure_is_structural(failure) for failure in failures):
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the WorldFoam framegroup16 promotion gate: benchmark-environment preflight, "
            "train/eval, then reference-artifact verifier."
        )
    )
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--frame-counts", default=DEFAULT_FRAME_COUNTS)
    parser.add_argument("--render-size", type=int, default=64)
    parser.add_argument("--site-count", type=int, default=24)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--tape-mode", default=DEFAULT_TAPE_MODE)
    parser.add_argument("--endpoint-record-source", default="gate4-affine")
    parser.add_argument("--optimizer-mode", default="manual-vjp")
    parser.add_argument("--reference-artifact", type=Path, default=None)
    parser.add_argument(
        "--no-reference-artifact",
        action="store_true",
        help="Run the verifier without a reference artifact. Required for custom frame-count dry-runs without a matching saved reference.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--partial-out-json", type=Path, default=None)
    parser.add_argument("--verify-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--allow-overwrite-artifacts",
        action="store_true",
        help="Allow pre-existing train/partial/verify artifacts for the selected run id. Default is fail-closed to avoid stale promotion evidence.",
    )
    parser.add_argument("--wait-for-benchmark-environment-ok", action="store_true")
    parser.add_argument("--wait-timeout-s", type=float, default=3600.0)
    parser.add_argument("--wait-interval-s", type=float, default=30.0)
    parser.add_argument(
        "--stable-preflight-checks",
        type=int,
        default=DEFAULT_STABLE_PREFLIGHT_CHECKS,
        help="Require this many consecutive successful benchmark preflight checks before train/eval launches.",
    )
    parser.add_argument(
        "--max-promotion-attempts",
        type=int,
        default=1,
        help=(
            "Retry the promotion with fresh attempt-suffixed artifacts when the verifier fails because "
            "the produced artifact was contaminated. Clean verifier regressions still fail immediately."
        ),
    )
    parser.add_argument("--experimental-native-cut-prep-delta", action="store_true")
    parser.add_argument("--experimental-native-sorted-delta", action="store_true")
    parser.add_argument("--experimental-minimal-packed-delta-device", action="store_true")
    parser.add_argument("--experimental-kernel-order-packed-delta-device", action="store_true")
    parser.add_argument("--experimental-smallrun16-packed-delta", action="store_true")
    parser.add_argument("--experimental-launch-only-packed-delta", action="store_true")
    parser.add_argument("--experimental-unchecked-launch-only-packed-delta", action="store_true")
    parser.add_argument("--experimental-reduce32-launch-only-packed-delta", action="store_true")
    parser.add_argument("--experimental-rowselect32-launch-only-packed-delta", action="store_true")
    parser.add_argument("--experimental-rowdesc-launch-only-packed-delta", action="store_true")
    parser.add_argument("--experimental-rowdesc32-launch-only-packed-delta", action="store_true")
    parser.add_argument("--experimental-cpu-rebase-delta", action="store_true")
    parser.add_argument("--experimental-native-pack-records", action="store_true")
    parser.add_argument("--experimental-native-emitted-pack-records", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = str(args.run_id)
    out_json = args.out_json or (RESULTS_DIR / f"{run_id}.json")
    partial_out_json = args.partial_out_json or (RESULTS_DIR / f"{run_id}.partial.json")
    verify_json = args.verify_json or (RESULTS_DIR / f"{run_id}.reference_verify.json")
    summary_json = args.summary_json or (RESULTS_DIR / f"{run_id}.promotion_summary.json")
    python = str(_repo_python())
    reference_explicit = args.reference_artifact is not None
    reference_artifact = None if bool(args.no_reference_artifact) else (args.reference_artifact or DEFAULT_REFERENCE)
    max_attempts = max(1, int(args.max_promotion_attempts))
    attempt_paths = _attempt_artifact_paths(
        out_json,
        partial_out_json,
        verify_json,
        max_attempts=max_attempts,
    )

    preflight_cmd = [
        python,
        str(TRAIN_EVAL),
        "--benchmark-environment-check-only",
    ]
    native_variant_flags = _enabled_native_variant_flags(args)
    variant_flags = _enabled_train_variant_flags(args)

    def train_cmd_for(attempt_out_json: Path, attempt_partial_out_json: Path) -> list[str]:
        return [
            python,
            str(TRAIN_EVAL),
            "--require-benchmark-environment-ok",
            "--tape-mode",
            str(args.tape_mode),
            "--endpoint-record-source",
            str(args.endpoint_record_source),
            "--frame-counts",
            str(args.frame_counts),
            "--render-size",
            str(args.render_size),
            "--site-count",
            str(args.site_count),
            "--optimizer-mode",
            str(args.optimizer_mode),
            "--steps",
            str(args.steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--out-json",
            str(attempt_out_json),
            "--partial-out-json",
            str(attempt_partial_out_json),
            *variant_flags,
        ]

    native_packed_extension_verify_cmd = (
        [python, str(NATIVE_PACKED_EXTENSION_VERIFY)] if _requires_native_packed_extension_verify(args) else None
    )

    def verify_cmd_for(attempt_out_json: Path, attempt_verify_json: Path) -> list[str]:
        command = [
            python,
            str(VERIFY),
            str(attempt_out_json),
            "--expected-frames",
            str(args.frame_counts),
            "--expected-tape-mode",
            str(args.tape_mode),
            "--out-json",
            str(attempt_verify_json),
            *_enabled_expected_payload_bools(args),
        ]
        if reference_artifact is not None:
            command[3:3] = ["--reference-artifact", str(reference_artifact)]
        return command

    first_out_json, first_partial_out_json, first_verify_json = attempt_paths[0]
    train_cmd = train_cmd_for(first_out_json, first_partial_out_json)
    verify_cmd = verify_cmd_for(first_out_json, first_verify_json)

    summary: dict[str, object] = {
        "run_id": run_id,
        "out_json": str(out_json),
        "partial_out_json": str(partial_out_json),
        "verify_json": str(verify_json),
        "reference_artifact": str(reference_artifact) if reference_artifact is not None else None,
        "reference_artifact_explicit": bool(reference_explicit),
        "no_reference_artifact": bool(args.no_reference_artifact),
        "preflight_command": preflight_cmd,
        "native_packed_extension_verify_command": native_packed_extension_verify_cmd,
        "train_command": train_cmd,
        "verify_command": verify_cmd,
        "native_variant_flags": variant_flags,
        "native_extension_variant_flags": native_variant_flags,
        "dry_run": bool(args.dry_run),
        "allow_overwrite_artifacts": bool(args.allow_overwrite_artifacts),
        "wait_for_benchmark_environment_ok": bool(args.wait_for_benchmark_environment_ok),
        "wait_timeout_s": float(args.wait_timeout_s),
        "wait_interval_s": float(args.wait_interval_s),
        "stable_preflight_checks": int(args.stable_preflight_checks),
        "max_promotion_attempts": max_attempts,
        "attempt_artifacts": [
            {
                "attempt_index": attempt_index,
                "out_json": str(attempt_out),
                "partial_out_json": str(attempt_partial),
                "verify_json": str(attempt_verify),
            }
            for attempt_index, (attempt_out, attempt_partial, attempt_verify) in enumerate(attempt_paths, 1)
        ],
    }
    config_failures: list[str] = []
    try:
        expected_frames = _parse_frame_counts(str(args.frame_counts))
    except ValueError as exc:
        config_failures.append(str(exc))
        expected_frames = tuple()
    if bool(args.no_reference_artifact) and reference_explicit:
        config_failures.append("--no-reference-artifact cannot be combined with --reference-artifact")
    if bool(args.experimental_minimal_packed_delta_device) and bool(args.experimental_kernel_order_packed_delta_device):
        config_failures.append(
            "--experimental-minimal-packed-delta-device and "
            "--experimental-kernel-order-packed-delta-device are mutually exclusive"
        )
    if bool(args.experimental_unchecked_launch_only_packed_delta) and not bool(args.experimental_launch_only_packed_delta):
        config_failures.append(
            "--experimental-unchecked-launch-only-packed-delta requires --experimental-launch-only-packed-delta"
        )
    if bool(args.experimental_reduce32_launch_only_packed_delta) and not bool(args.experimental_launch_only_packed_delta):
        config_failures.append(
            "--experimental-reduce32-launch-only-packed-delta requires --experimental-launch-only-packed-delta"
        )
    if bool(args.experimental_rowselect32_launch_only_packed_delta) and not bool(
        args.experimental_launch_only_packed_delta
    ):
        config_failures.append(
            "--experimental-rowselect32-launch-only-packed-delta requires --experimental-launch-only-packed-delta"
        )
    if bool(args.experimental_rowdesc_launch_only_packed_delta) and not bool(args.experimental_launch_only_packed_delta):
        config_failures.append(
            "--experimental-rowdesc-launch-only-packed-delta requires --experimental-launch-only-packed-delta"
        )
    if bool(args.experimental_rowselect32_launch_only_packed_delta) and bool(
        args.experimental_reduce32_launch_only_packed_delta
    ):
        config_failures.append(
            "--experimental-rowselect32-launch-only-packed-delta cannot be combined with "
            "--experimental-reduce32-launch-only-packed-delta"
        )
    if bool(args.experimental_rowselect32_launch_only_packed_delta) and bool(
        args.experimental_rowdesc_launch_only_packed_delta
    ):
        config_failures.append(
            "--experimental-rowselect32-launch-only-packed-delta cannot be combined with "
            "--experimental-rowdesc-launch-only-packed-delta"
        )
    if bool(args.experimental_reduce32_launch_only_packed_delta) and bool(args.experimental_rowdesc_launch_only_packed_delta):
        config_failures.append(
            "--experimental-reduce32-launch-only-packed-delta cannot be combined with "
            "--experimental-rowdesc-launch-only-packed-delta"
        )
    if bool(args.experimental_rowdesc32_launch_only_packed_delta) and not bool(
        args.experimental_rowdesc_launch_only_packed_delta
    ):
        config_failures.append(
            "--experimental-rowdesc32-launch-only-packed-delta requires "
            "--experimental-rowdesc-launch-only-packed-delta"
        )
    output_artifact_paths = tuple(path for paths in attempt_paths for path in paths)
    preexisting_output_artifacts = [str(path) for path in output_artifact_paths if path.exists()]
    if preexisting_output_artifacts:
        summary["preexisting_output_artifacts"] = preexisting_output_artifacts
        if not bool(args.allow_overwrite_artifacts) and not bool(args.dry_run):
            config_failures.append(
                "pre-existing output artifacts would make promotion evidence ambiguous; "
                "choose a new --run-id, remove the stale artifacts, or pass --allow-overwrite-artifacts: "
                + ", ".join(preexisting_output_artifacts)
            )
    if reference_artifact is not None and expected_frames:
        try:
            reference_frames = _artifact_frame_counts(reference_artifact)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            config_failures.append(f"{reference_artifact}: could not validate reference frames: {exc}")
        else:
            summary["reference_artifact_frames"] = list(reference_frames)
            missing_frames = tuple(frame for frame in expected_frames if frame not in reference_frames)
            if missing_frames:
                if not reference_explicit:
                    config_failures.append(
                        f"requested frame counts {expected_frames} are not covered by the default reference "
                        f"{reference_frames}; pass --reference-artifact with matching frames or --no-reference-artifact"
                    )
                else:
                    config_failures.append(
                        f"reference frame counts {reference_frames} do not cover requested {expected_frames}; "
                        f"missing {missing_frames}"
                    )
    if config_failures:
        summary["status"] = "config_failed"
        summary["config_failures"] = config_failures
        _write_summary(summary_json, summary)
        for failure in config_failures:
            print(f"[promotion_gate] {failure}", file=sys.stderr, flush=True)
        return 2

    if native_packed_extension_verify_cmd is not None:
        native_verify_status, native_verify_result = _run_json_command(
            native_packed_extension_verify_cmd,
            dry_run=bool(args.dry_run),
        )
        summary["native_packed_extension_verify_status"] = native_verify_status
        if native_verify_result is not None:
            summary["native_packed_extension_verify_result"] = native_verify_result
        native_verify_failures = _native_packed_verify_failures(
            native_verify_result,
            dry_run=bool(args.dry_run),
        )
        if native_verify_failures:
            summary["native_packed_extension_verify_failures"] = native_verify_failures
        if native_verify_status != 0 or native_verify_failures:
            summary["status"] = "native_packed_extension_verify_failed"
            _write_summary(summary_json, summary)
            return native_verify_status if native_verify_status != 0 else 2

    attempts: list[dict[str, object]] = []
    summary["attempts"] = attempts
    for attempt_index, (attempt_out_json, attempt_partial_out_json, attempt_verify_json) in enumerate(
        attempt_paths,
        1,
    ):
        attempt_train_cmd = train_cmd_for(attempt_out_json, attempt_partial_out_json)
        attempt_verify_cmd = verify_cmd_for(attempt_out_json, attempt_verify_json)
        attempt_summary: dict[str, object] = {
            "attempt_index": attempt_index,
            "out_json": str(attempt_out_json),
            "partial_out_json": str(attempt_partial_out_json),
            "verify_json": str(attempt_verify_json),
            "train_command": attempt_train_cmd,
            "verify_command": attempt_verify_cmd,
        }
        attempts.append(attempt_summary)
        summary["current_attempt_index"] = attempt_index
        summary["status"] = "preflight_pending"
        _write_summary(summary_json, summary)

        preflight_status, preflight_attempts = _run_preflight(
            preflight_cmd,
            dry_run=bool(args.dry_run),
            wait=bool(args.wait_for_benchmark_environment_ok),
            timeout_s=float(args.wait_timeout_s),
            interval_s=float(args.wait_interval_s),
            stable_checks=int(args.stable_preflight_checks),
            summary=summary,
            summary_path=summary_json,
        )
        attempt_summary["preflight_status"] = preflight_status
        attempt_summary["preflight_attempts"] = preflight_attempts
        attempt_summary.update(
            _preflight_progress_fields(
                preflight_attempts,
                required_successes=int(args.stable_preflight_checks),
            )
        )
        summary["preflight_status"] = preflight_status
        summary["preflight_attempts"] = preflight_attempts
        summary.update(
            _preflight_progress_fields(
                preflight_attempts,
                required_successes=int(args.stable_preflight_checks),
            )
        )
        if preflight_status != 0:
            summary["status"] = "preflight_failed"
            summary["preflight_failure_reason"] = _preflight_failure_reason(
                preflight_status,
                preflight_attempts,
                required_successes=int(args.stable_preflight_checks),
            )
            _write_summary(summary_json, summary)
            return preflight_status

        train_status = _run(attempt_train_cmd, dry_run=bool(args.dry_run))
        attempt_summary["train_status"] = train_status
        summary["train_status"] = train_status
        if train_status != 0:
            summary["status"] = "train_failed"
            _write_summary(summary_json, summary)
            return train_status

        verify_status = _run(attempt_verify_cmd, dry_run=bool(args.dry_run))
        attempt_summary["verify_status"] = verify_status
        summary["verify_status"] = verify_status
        verify_brief = _load_verify_brief(attempt_verify_json)
        if verify_brief is not None:
            attempt_summary["verify_result"] = verify_brief
            summary["verify_result"] = verify_brief
        if verify_status == 0:
            summary["status"] = "ok"
            _write_summary(summary_json, summary)
            return 0
        if _verify_brief_is_retryable_contamination(verify_brief) and attempt_index < max_attempts:
            attempt_summary["retry_reason"] = "verify_contamination"
            summary["status"] = "retrying_after_verify_contamination"
            _write_summary(summary_json, summary)
            continue
        summary["status"] = "verify_failed"
        _write_summary(summary_json, summary)
        return verify_status

    summary["status"] = "verify_failed"
    _write_summary(summary_json, summary)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
