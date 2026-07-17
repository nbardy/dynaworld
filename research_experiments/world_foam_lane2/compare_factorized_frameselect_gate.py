#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import run_framegroup16_promotion_gate as promotion_gate


REGULAR_MODE = "owner-run-delta-packed-factorized-recompute-fused-mse-nomid"
FRAMESELECT_MODE = "owner-run-delta-packed-factorized-frameselect-recompute-fused-mse-nomid"
FRAMEBITMASK_MODE = "owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid"
DEFAULT_CANDIDATE_LABELS = ("frameselect",)
CANDIDATE_MODES = {
    "frameselect": FRAMESELECT_MODE,
    "framebitmask": FRAMEBITMASK_MODE,
}
DEFAULT_FRAME_COUNTS = "2,4,8,16"
DEFAULT_RENDER_SIZE = 16
DEFAULT_SITE_COUNT = 8
DEFAULT_STEPS = 2
DEFAULT_WARMUP_STEPS = 1
DEFAULT_MAX_TOTAL_RATIO = 1.10
DEFAULT_MAX_BACKWARD_RATIO = 1.10
DEFAULT_MAX_STORAGE_RATIO = 1.00
DEFAULT_MAX_COMPARISON_ATTEMPTS = 1


class _ComparisonInterrupted(RuntimeError):
    pass


def _default_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_factorized_frameselect_compare_%H%M%S")


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clear_mode_result_fields(summary: dict[str, Any], label: str) -> None:
    for suffix in (
        "preflight_status",
        "preflight_attempts",
        "preflight_failure_reason",
        "train_status",
        "artifact_status",
        "benchmark_environment_status",
        "artifact_clean",
        "artifact_missing_after_train_status",
    ):
        summary.pop(f"{label}_{suffix}", None)


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _install_interrupt_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def _handle(signum: int, _frame: Any) -> None:
        raise _ComparisonInterrupted(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle)
    return previous


def _restore_interrupt_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _benchmark_environment_status(payload: dict[str, Any]) -> str | None:
    environment = payload.get("benchmark_environment")
    if not isinstance(environment, dict):
        return None
    status = environment.get("status")
    return str(status) if isinstance(status, str) else None


def _artifact_clean(payload: dict[str, Any]) -> bool:
    return payload.get("status") == "ok" and _benchmark_environment_status(payload) == "background"


def _artifact_contaminated(payload: dict[str, Any]) -> bool:
    return _benchmark_environment_status(payload) != "background"


def _retryable_train_failure(train_status: int, payload: dict[str, Any] | None) -> bool:
    if payload is not None:
        return _artifact_contaminated(payload)
    # train_eval_owner_run_tape exits 2 before writing out_json when the
    # child-side --require-benchmark-environment-ok check catches a contender
    # after the parent preflight but before tape construction.
    return train_status == 2


def _rows_by_frame(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("artifact rows must be a list")
    by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("frame_count"), int):
            by_frame[int(row["frame_count"])] = row
    if not by_frame:
        raise ValueError("artifact rows did not contain frame_count entries")
    return by_frame


def _median_ms(row: dict[str, Any], group: str) -> float | None:
    step_summary = row.get("step_summary")
    if not isinstance(step_summary, dict):
        return None
    group_summary = step_summary.get(group)
    if not isinstance(group_summary, dict):
        return None
    median_s = group_summary.get("median_s")
    return float(median_s) * 1000.0 if isinstance(median_s, (int, float)) else None


def _number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _max_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def compare_candidate_payloads(
    regular: dict[str, Any],
    candidate: dict[str, Any],
    *,
    candidate_label: str,
    max_total_ratio: float,
    max_backward_ratio: float,
    max_storage_ratio: float,
) -> dict[str, Any]:
    regular_rows = _rows_by_frame(regular)
    candidate_rows = _rows_by_frame(candidate)
    frames = tuple(sorted(set(regular_rows) & set(candidate_rows)))
    missing_regular = tuple(sorted(set(candidate_rows) - set(regular_rows)))
    missing_candidate = tuple(sorted(set(regular_rows) - set(candidate_rows)))
    rows: dict[str, Any] = {}
    total_ratios: list[float | None] = []
    backward_ratios: list[float | None] = []
    storage_ratios: list[float | None] = []
    topology_ratios: list[float | None] = []
    noncoeff_ratios: list[float | None] = []

    for frame in frames:
        regular_row = regular_rows[frame]
        candidate_row = candidate_rows[frame]
        regular_total_ms = _median_ms(regular_row, "total")
        candidate_total_ms = _median_ms(candidate_row, "total")
        regular_backward_ms = _median_ms(regular_row, "backward")
        candidate_backward_ms = _median_ms(candidate_row, "backward")
        regular_storage = _number(regular_row, "train_selected_tape_schema_storage_bytes")
        candidate_storage = _number(candidate_row, "train_selected_tape_schema_storage_bytes")
        regular_topology = _number(regular_row, "train_selected_tape_schema_topology_storage_bytes")
        candidate_topology = _number(candidate_row, "train_selected_tape_schema_topology_storage_bytes")
        regular_noncoeff = _number(regular_row, "train_selected_tape_mps_resident_noncoeff_storage_bytes")
        candidate_noncoeff = _number(
            candidate_row,
            "train_selected_tape_mps_resident_noncoeff_storage_bytes",
        )
        regular_psnr = _number(regular_row, "final_train_psnr")
        candidate_psnr = _number(candidate_row, "final_train_psnr")

        total_ratio = _ratio(candidate_total_ms, regular_total_ms)
        backward_ratio = _ratio(candidate_backward_ms, regular_backward_ms)
        storage_ratio = _ratio(candidate_storage, regular_storage)
        topology_ratio = _ratio(candidate_topology, regular_topology)
        noncoeff_ratio = _ratio(candidate_noncoeff, regular_noncoeff)
        total_ratios.append(total_ratio)
        backward_ratios.append(backward_ratio)
        storage_ratios.append(storage_ratio)
        topology_ratios.append(topology_ratio)
        noncoeff_ratios.append(noncoeff_ratio)

        rows[str(frame)] = {
            "regular": {
                "status": regular_row.get("status"),
                "total_median_ms": regular_total_ms,
                "backward_median_ms": regular_backward_ms,
                "schema_storage_bytes": regular_storage,
                "topology_storage_bytes": regular_topology,
                "mps_resident_noncoeff_storage_bytes": regular_noncoeff,
                "train_psnr": regular_psnr,
            },
            candidate_label: {
                "status": candidate_row.get("status"),
                "total_median_ms": candidate_total_ms,
                "backward_median_ms": candidate_backward_ms,
                "schema_storage_bytes": candidate_storage,
                "topology_storage_bytes": candidate_topology,
                "mps_resident_noncoeff_storage_bytes": candidate_noncoeff,
                "train_psnr": candidate_psnr,
            },
            f"{candidate_label}_over_regular": {
                "total_median_ratio": total_ratio,
                "backward_median_ratio": backward_ratio,
                "schema_storage_ratio": storage_ratio,
                "topology_storage_ratio": topology_ratio,
                "mps_resident_noncoeff_storage_ratio": noncoeff_ratio,
                "train_psnr_delta": (
                    candidate_psnr - regular_psnr
                    if candidate_psnr is not None and regular_psnr is not None
                    else None
                ),
            },
        }

    max_total = _max_present(total_ratios)
    max_backward = _max_present(backward_ratios)
    max_storage = _max_present(storage_ratios)
    clean_speedscale_artifact = _artifact_clean(regular) and _artifact_clean(candidate)
    failures: list[str] = []
    if missing_regular:
        failures.append(f"regular artifact is missing frame rows: {missing_regular}")
    if missing_candidate:
        failures.append(f"{candidate_label} artifact is missing frame rows: {missing_candidate}")
    if not clean_speedscale_artifact:
        failures.append(
            "benchmark artifacts are not clean: "
            f"regular={_benchmark_environment_status(regular)!r}, "
            f"{candidate_label}={_benchmark_environment_status(candidate)!r}"
        )
    if max_total is not None and max_total > max_total_ratio:
        failures.append(f"{candidate_label} total median ratio {max_total:.3f} exceeds {max_total_ratio:.3f}")
    if max_backward is not None and max_backward > max_backward_ratio:
        failures.append(f"{candidate_label} backward median ratio {max_backward:.3f} exceeds {max_backward_ratio:.3f}")
    if max_storage is not None and max_storage > max_storage_ratio:
        failures.append(f"{candidate_label} schema storage ratio {max_storage:.3f} exceeds {max_storage_ratio:.3f}")

    recommendation = f"{candidate_label}_candidate"
    if not clean_speedscale_artifact:
        recommendation = "rerun_clean"
    elif failures:
        recommendation = "keep_regular_or_fork_again"

    return {
        "status": "ok" if not failures else "failed",
        "recommendation": recommendation,
        "clean_speedscale_artifact": clean_speedscale_artifact,
        "regular_status": regular.get("status"),
        f"{candidate_label}_status": candidate.get("status"),
        "regular_benchmark_environment_status": _benchmark_environment_status(regular),
        f"{candidate_label}_benchmark_environment_status": _benchmark_environment_status(candidate),
        "frames_compared": list(frames),
        "missing_regular_frames": list(missing_regular),
        f"missing_{candidate_label}_frames": list(missing_candidate),
        "max_total_median_ratio": max_total,
        "max_backward_median_ratio": max_backward,
        "max_schema_storage_ratio": max_storage,
        "max_topology_storage_ratio": _max_present(topology_ratios),
        "max_mps_resident_noncoeff_storage_ratio": _max_present(noncoeff_ratios),
        "ratio_thresholds": {
            "max_total_ratio": max_total_ratio,
            "max_backward_ratio": max_backward_ratio,
            "max_storage_ratio": max_storage_ratio,
        },
        "failures": failures,
        "rows": rows,
    }


def compare_payloads(
    regular: dict[str, Any],
    frameselect: dict[str, Any],
    *,
    max_total_ratio: float,
    max_backward_ratio: float,
    max_storage_ratio: float,
) -> dict[str, Any]:
    return compare_candidate_payloads(
        regular,
        frameselect,
        candidate_label="frameselect",
        max_total_ratio=max_total_ratio,
        max_backward_ratio=max_backward_ratio,
        max_storage_ratio=max_storage_ratio,
    )


def compare_candidate_set(
    regular: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    *,
    max_total_ratio: float,
    max_backward_ratio: float,
    max_storage_ratio: float,
) -> dict[str, Any]:
    candidate_comparisons = {
        label: compare_candidate_payloads(
            regular,
            payload,
            candidate_label=label,
            max_total_ratio=max_total_ratio,
            max_backward_ratio=max_backward_ratio,
            max_storage_ratio=max_storage_ratio,
        )
        for label, payload in candidates.items()
    }
    passing = [
        label
        for label, comparison in candidate_comparisons.items()
        if comparison.get("status") == "ok"
    ]
    if passing:
        best_label = min(
            passing,
            key=lambda label: (
                candidate_comparisons[label].get("max_total_median_ratio") or float("inf"),
                candidate_comparisons[label].get("max_backward_median_ratio") or float("inf"),
                candidate_comparisons[label].get("max_schema_storage_ratio") or float("inf"),
            ),
        )
        return {
            "status": "ok",
            "recommendation": f"{best_label}_candidate",
            "candidate_labels": list(candidates),
            "passing_candidates": passing,
            "best_candidate": best_label,
            "candidate_comparisons": candidate_comparisons,
        }
    recommendation = "rerun_clean" if any(
        comparison.get("recommendation") == "rerun_clean"
        for comparison in candidate_comparisons.values()
    ) else "keep_regular_or_fork_again"
    return {
        "status": "failed",
        "recommendation": recommendation,
        "candidate_labels": list(candidates),
        "passing_candidates": [],
        "best_candidate": None,
        "candidate_comparisons": candidate_comparisons,
    }


def _parse_candidate_labels(value: str) -> tuple[str, ...]:
    labels = tuple(dict.fromkeys(label.strip() for label in value.split(",") if label.strip()))
    if not labels:
        raise argparse.ArgumentTypeError("candidate label list must not be empty")
    unknown = sorted(set(labels) - set(CANDIDATE_MODES))
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown candidate label(s): "
            + ", ".join(unknown)
            + f"; expected one or more of {', '.join(sorted(CANDIDATE_MODES))}"
        )
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and compare regular factorized WorldFoam against frame-select factorized WorldFoam."
    )
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--frame-counts", default=DEFAULT_FRAME_COUNTS)
    parser.add_argument("--render-size", type=int, default=DEFAULT_RENDER_SIZE)
    parser.add_argument("--site-count", type=int, default=DEFAULT_SITE_COUNT)
    parser.add_argument("--near", type=float, default=0.0)
    parser.add_argument("--far", type=float, default=3.5)
    parser.add_argument("--density", type=float, default=8.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--optimizer-mode", choices=("manual-vjp", "autograd"), default="manual-vjp")
    parser.add_argument("--endpoint-record-source", choices=("slow-owner-run", "gate4-affine"), default="slow-owner-run")
    parser.add_argument("--defer-heldout-device", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--regular-out-json", type=Path)
    parser.add_argument("--frameselect-out-json", type=Path)
    parser.add_argument("--framebitmask-out-json", type=Path)
    parser.add_argument(
        "--accepted-regular-json",
        type=Path,
        help="Reuse an already-clean regular artifact instead of re-running regular.",
    )
    parser.add_argument(
        "--accepted-frameselect-json",
        type=Path,
        help="Reuse an already-clean frame-select artifact instead of re-running frame-select.",
    )
    parser.add_argument(
        "--accepted-framebitmask-json",
        type=Path,
        help="Reuse an already-clean frame-bitmask artifact instead of re-running frame-bitmask.",
    )
    parser.add_argument(
        "--include-framebitmask",
        action="store_true",
        help="Also run and compare the compact per-track frame-bitmask selector candidate.",
    )
    parser.add_argument(
        "--candidate-labels",
        type=_parse_candidate_labels,
        help=(
            "Comma-separated candidate labels to run/compare. Overrides the default "
            "frameselect plus optional --include-framebitmask selection."
        ),
    )
    parser.add_argument("--allow-overwrite-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-benchmark-environment-ok", action="store_true")
    parser.add_argument("--wait-timeout-s", type=float, default=3600.0)
    parser.add_argument("--wait-interval-s", type=float, default=30.0)
    parser.add_argument(
        "--stable-preflight-checks",
        type=int,
        default=promotion_gate.DEFAULT_STABLE_PREFLIGHT_CHECKS,
        help="Require this many consecutive clean preflight snapshots before each train/eval command.",
    )
    parser.add_argument("--max-total-regression-ratio", type=float, default=DEFAULT_MAX_TOTAL_RATIO)
    parser.add_argument("--max-backward-regression-ratio", type=float, default=DEFAULT_MAX_BACKWARD_RATIO)
    parser.add_argument("--max-storage-regression-ratio", type=float, default=DEFAULT_MAX_STORAGE_RATIO)
    parser.add_argument(
        "--max-comparison-attempts",
        type=int,
        default=DEFAULT_MAX_COMPARISON_ATTEMPTS,
        help=(
            "Retry each individual mode when a train artifact ends with a contended "
            "benchmark environment. Each retry writes a distinct per-mode attempt artifact."
        ),
    )
    parser.add_argument(
        "--continue-after-contaminated-artifact",
        action="store_true",
        help="Continue to the next mode even if a train artifact ends with a contended benchmark environment.",
    )
    return parser


def _train_command(args: argparse.Namespace, *, mode: str, out_json: Path) -> list[str]:
    cmd = [
        str(promotion_gate._repo_python()),
        str(promotion_gate.TRAIN_EVAL),
        "--frame-counts",
        str(args.frame_counts),
        "--render-size",
        str(args.render_size),
        "--site-count",
        str(args.site_count),
        "--near",
        str(args.near),
        "--far",
        str(args.far),
        "--density",
        str(args.density),
        "--invalid-epsilon",
        str(args.invalid_epsilon),
        "--transmittance-threshold",
        str(args.transmittance_threshold),
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--optimizer-mode",
        str(args.optimizer_mode),
        "--tape-mode",
        mode,
        "--endpoint-record-source",
        str(args.endpoint_record_source),
        "--require-benchmark-environment-ok",
        "--out-json",
        str(out_json),
    ]
    if bool(args.defer_heldout_device):
        cmd.append("--defer-heldout-device")
    return cmd


def _candidate_labels(args: argparse.Namespace) -> tuple[str, ...]:
    if args.candidate_labels is not None:
        return tuple(args.candidate_labels)
    labels = list(DEFAULT_CANDIDATE_LABELS)
    if bool(args.include_framebitmask) and "framebitmask" not in labels:
        labels.append("framebitmask")
    return tuple(labels)


def _attempt_path(path: Path, attempt_index: int, max_attempts: int) -> Path:
    if max_attempts <= 1:
        return path
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}.attempt{attempt_index}{suffix}")


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Path]]:
    run_id = str(args.run_id)
    summary_json = args.summary_json or (
        promotion_gate.RESULTS_DIR / f"{run_id}.factorized_frameselect_compare_summary.json"
    )
    regular_out = args.regular_out_json or (
        promotion_gate.RESULTS_DIR / f"{run_id}.regular_factorized.json"
    )
    candidate_outs = {
        "frameselect": args.frameselect_out_json
        or promotion_gate.RESULTS_DIR / f"{run_id}.frameselect_factorized.json"
    }
    if "framebitmask" in _candidate_labels(args):
        candidate_outs["framebitmask"] = (
            args.framebitmask_out_json
            or promotion_gate.RESULTS_DIR / f"{run_id}.framebitmask_factorized.json"
        )
    return summary_json, regular_out, candidate_outs


def _check_output_paths(paths: tuple[Path, ...], *, allow_overwrite: bool, dry_run: bool) -> list[str]:
    if allow_overwrite or dry_run:
        return []
    existing = [str(path) for path in paths if path.exists()]
    if not existing:
        return []
    return [
        "pre-existing output artifacts would make comparison evidence ambiguous; "
        "choose a new --run-id, remove stale artifacts, or pass --allow-overwrite-artifacts: "
        + ", ".join(existing)
    ]


def _accepted_artifact_paths(args: argparse.Namespace, labels: tuple[str, ...]) -> dict[str, Path]:
    accepted: dict[str, Path] = {}
    if args.accepted_regular_json is not None:
        accepted["regular"] = args.accepted_regular_json
    if args.accepted_frameselect_json is not None:
        accepted["frameselect"] = args.accepted_frameselect_json
    if args.accepted_framebitmask_json is not None:
        accepted["framebitmask"] = args.accepted_framebitmask_json
    allowed = {"regular", *labels}
    return {label: path for label, path in accepted.items() if label in allowed}


def _check_accepted_artifact_paths(args: argparse.Namespace, labels: tuple[str, ...]) -> list[str]:
    failures: list[str] = []
    if args.accepted_framebitmask_json is not None and "framebitmask" not in labels:
        failures.append("--accepted-framebitmask-json requires --include-framebitmask")
    for label, path in _accepted_artifact_paths(args, labels).items():
        if not path.exists():
            failures.append(f"--accepted-{label}-json does not exist: {path}")
    return failures


def _preflight_command() -> list[str]:
    return [
        str(promotion_gate._repo_python()),
        str(promotion_gate.TRAIN_EVAL),
        "--benchmark-environment-check-only",
    ]


def _run_mode(
    args: argparse.Namespace,
    *,
    label: str,
    mode: str,
    out_json: Path,
    summary: dict[str, Any],
    summary_json: Path,
) -> tuple[int, dict[str, Any] | None]:
    _clear_mode_result_fields(summary, label)
    preflight_cmd = _preflight_command()
    preflight_status, preflight_attempts = promotion_gate._run_preflight(
        preflight_cmd,
        dry_run=bool(args.dry_run),
        wait=bool(args.wait_for_benchmark_environment_ok),
        timeout_s=float(args.wait_timeout_s),
        interval_s=float(args.wait_interval_s),
        stable_checks=int(args.stable_preflight_checks),
        summary=summary,
        summary_path=summary_json,
    )
    summary[f"{label}_preflight_status"] = preflight_status
    summary[f"{label}_preflight_attempts"] = preflight_attempts
    if preflight_status != 0:
        summary["status"] = f"preflight_failed_before_{label}"
        summary[f"{label}_preflight_failure_reason"] = promotion_gate._preflight_failure_reason(
            preflight_status,
            preflight_attempts,
            required_successes=int(args.stable_preflight_checks),
        )
        _write_summary(summary_json, summary)
        return preflight_status, None

    train_cmd = _train_command(args, mode=mode, out_json=out_json)
    summary[f"{label}_train_command"] = train_cmd
    summary["status"] = f"running_{label}"
    _write_summary(summary_json, summary)
    train_status = promotion_gate._run(train_cmd, dry_run=bool(args.dry_run))
    summary[f"{label}_train_status"] = train_status
    if train_status != 0:
        payload = _load_payload(out_json) if out_json.exists() else None
        if payload is not None:
            summary[f"{label}_artifact_status"] = payload.get("status")
            summary[f"{label}_benchmark_environment_status"] = _benchmark_environment_status(payload)
            if _artifact_contaminated(payload) and not bool(args.continue_after_contaminated_artifact):
                summary["status"] = f"{label}_artifact_contaminated"
                summary[f"{label}_artifact_clean"] = False
                _write_summary(summary_json, summary)
                return 2, payload
        elif train_status == 2:
            summary["status"] = f"{label}_start_environment_contended"
            summary[f"{label}_artifact_missing_after_train_status"] = train_status
            _write_summary(summary_json, summary)
            return train_status, None
        summary["status"] = f"{label}_train_failed"
        _write_summary(summary_json, summary)
        return train_status, payload
    if bool(args.dry_run):
        _write_summary(summary_json, summary)
        return 0, None

    payload = _load_payload(out_json)
    summary[f"{label}_artifact_status"] = payload.get("status")
    summary[f"{label}_benchmark_environment_status"] = _benchmark_environment_status(payload)
    if not _artifact_clean(payload) and not bool(args.continue_after_contaminated_artifact):
        summary["status"] = f"{label}_artifact_contaminated"
        summary[f"{label}_artifact_clean"] = False
        _write_summary(summary_json, summary)
        return 2, payload
    _write_summary(summary_json, summary)
    return 0, payload


def _retry_reason(label: str, payload: dict[str, Any] | None) -> str:
    return f"{label}_artifact_contaminated" if payload is not None else f"{label}_start_environment_contended"


def _record_attempt_payload_status(
    attempt_summary: dict[str, Any],
    label: str,
    payload: dict[str, Any] | None,
) -> None:
    if payload is None:
        return
    attempt_summary[f"{label}_benchmark_environment_status"] = _benchmark_environment_status(payload)


def _run_mode_until_clean(
    args: argparse.Namespace,
    *,
    label: str,
    mode: str,
    out_base: Path,
    max_attempts: int,
    summary: dict[str, Any],
    summary_json: Path,
    attempts: list[dict[str, Any]],
) -> tuple[int, dict[str, Any] | None]:
    for attempt_index in range(1, max_attempts + 1):
        out_json = _attempt_path(out_base, attempt_index, max_attempts)
        attempt_summary: dict[str, Any] = {
            "mode_label": label,
            "attempt_index": attempt_index,
            f"{label}_out_json": str(out_json),
            f"{label}_train_command": _train_command(args, mode=mode, out_json=out_json),
        }
        attempts.append(attempt_summary)
        summary["current_mode_label"] = label
        summary["current_attempt_index"] = attempt_index
        summary["status"] = f"{label}_attempt_pending"
        _write_summary(summary_json, summary)

        status, payload = _run_mode(
            args,
            label=label,
            mode=mode,
            out_json=out_json,
            summary=summary,
            summary_json=summary_json,
        )
        attempt_summary[f"{label}_status"] = status
        _record_attempt_payload_status(attempt_summary, label, payload)
        if status == 0:
            attempt_summary["accepted"] = True
            summary[f"{label}_accepted_attempt_index"] = attempt_index
            summary[f"{label}_accepted_out_json"] = str(out_json)
            _write_summary(summary_json, summary)
            return status, payload

        retryable = _retryable_train_failure(status, payload)
        if retryable and attempt_index < max_attempts:
            attempt_summary["retry_reason"] = _retry_reason(label, payload)
            summary["status"] = f"retrying_after_{attempt_summary['retry_reason']}"
            _write_summary(summary_json, summary)
            continue
        _write_summary(summary_json, summary)
        return status, payload

    summary["status"] = f"{label}_attempts_exhausted"
    _write_summary(summary_json, summary)
    return 2, None


def _reuse_accepted_artifact(
    args: argparse.Namespace,
    *,
    label: str,
    path: Path,
    summary: dict[str, Any],
    summary_json: Path,
) -> tuple[int, dict[str, Any] | None]:
    _clear_mode_result_fields(summary, label)
    summary["current_mode_label"] = label
    summary.pop("current_attempt_index", None)
    summary["status"] = f"reusing_{label}_artifact"
    summary[f"{label}_accepted_out_json"] = str(path)
    summary[f"{label}_accepted_artifact_source"] = "input"
    _write_summary(summary_json, summary)

    payload = _load_payload(path)
    summary[f"{label}_artifact_status"] = payload.get("status")
    summary[f"{label}_benchmark_environment_status"] = _benchmark_environment_status(payload)
    if not _artifact_clean(payload) and not bool(args.continue_after_contaminated_artifact):
        summary["status"] = f"{label}_accepted_artifact_not_clean"
        summary[f"{label}_artifact_clean"] = False
        _write_summary(summary_json, summary)
        return 2, payload
    summary[f"{label}_artifact_clean"] = _artifact_clean(payload)
    _write_summary(summary_json, summary)
    return 0, payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    max_attempts = max(1, int(args.max_comparison_attempts))
    candidate_labels = _candidate_labels(args)
    accepted_artifacts = _accepted_artifact_paths(args, candidate_labels)
    summary_json, regular_out, candidate_outs = _output_paths(args)
    first_regular_out = _attempt_path(regular_out, 1, max_attempts)
    regular_cmd = _train_command(args, mode=REGULAR_MODE, out_json=first_regular_out)
    candidate_cmds = {
        label: _train_command(
            args,
            mode=CANDIDATE_MODES[label],
            out_json=_attempt_path(candidate_outs[label], 1, max_attempts),
        )
        for label in candidate_labels
    }
    attempt_artifacts = [
        {
            "attempt_index": attempt_index,
            "regular_out_json": str(_attempt_path(regular_out, attempt_index, max_attempts)),
            **{
                f"{label}_out_json": str(_attempt_path(candidate_outs[label], attempt_index, max_attempts))
                for label in candidate_labels
            },
        }
        for attempt_index in range(1, max_attempts + 1)
    ]
    summary: dict[str, Any] = {
        "run_id": str(args.run_id),
        "status": "configured",
        "summary_json": str(summary_json),
        "regular_out_json": str(regular_out),
        **{f"{label}_out_json": str(candidate_outs[label]) for label in candidate_labels},
        "candidate_labels": list(candidate_labels),
        "max_comparison_attempts": max_attempts,
        "attempt_artifacts": attempt_artifacts,
        "preflight_command": _preflight_command(),
        "regular_train_command": regular_cmd,
        **{f"{label}_train_command": candidate_cmds[label] for label in candidate_labels},
        "dry_run": bool(args.dry_run),
        "wait_for_benchmark_environment_ok": bool(args.wait_for_benchmark_environment_ok),
        "wait_timeout_s": float(args.wait_timeout_s),
        "wait_interval_s": float(args.wait_interval_s),
        "stable_preflight_checks": int(args.stable_preflight_checks),
        "continue_after_contaminated_artifact": bool(args.continue_after_contaminated_artifact),
        "accepted_input_artifacts": {
            label: str(path) for label, path in accepted_artifacts.items()
        },
    }
    output_artifact_paths = [summary_json]
    for attempt in attempt_artifacts:
        output_artifact_paths.extend(
            [
                Path(str(attempt["regular_out_json"])),
                *(Path(str(attempt[f"{label}_out_json"])) for label in candidate_labels),
            ]
        )
    config_failures = _check_output_paths(
        tuple(output_artifact_paths),
        allow_overwrite=bool(args.allow_overwrite_artifacts),
        dry_run=bool(args.dry_run),
    )
    config_failures.extend(_check_accepted_artifact_paths(args, candidate_labels))
    if config_failures:
        summary["status"] = "config_failed"
        summary["config_failures"] = config_failures
        _write_summary(summary_json, summary)
        for failure in config_failures:
            print(f"[factorized_frameselect_gate] {failure}", file=sys.stderr, flush=True)
        return 2

    _write_summary(summary_json, summary)
    previous_handlers = _install_interrupt_handlers()
    try:
        attempts: list[dict[str, Any]] = []
        summary["attempts"] = attempts
        if "regular" in accepted_artifacts:
            regular_status, regular_payload = _reuse_accepted_artifact(
                args,
                label="regular",
                path=accepted_artifacts["regular"],
                summary=summary,
                summary_json=summary_json,
            )
        else:
            regular_status, regular_payload = _run_mode_until_clean(
                args,
                label="regular",
                mode=REGULAR_MODE,
                out_base=regular_out,
                max_attempts=max_attempts,
                summary=summary,
                summary_json=summary_json,
                attempts=attempts,
            )
        if regular_status != 0:
            return regular_status

        candidate_payloads: dict[str, dict[str, Any]] = {}
        for label in candidate_labels:
            if label in accepted_artifacts:
                candidate_status, candidate_payload = _reuse_accepted_artifact(
                    args,
                    label=label,
                    path=accepted_artifacts[label],
                    summary=summary,
                    summary_json=summary_json,
                )
            else:
                candidate_status, candidate_payload = _run_mode_until_clean(
                    args,
                    label=label,
                    mode=CANDIDATE_MODES[label],
                    out_base=candidate_outs[label],
                    max_attempts=max_attempts,
                    summary=summary,
                    summary_json=summary_json,
                    attempts=attempts,
                )
            if candidate_status != 0:
                return candidate_status
            if candidate_payload is not None:
                candidate_payloads[label] = candidate_payload

        if bool(args.dry_run):
            summary["status"] = "dry_run"
            _write_summary(summary_json, summary)
            return 0
        if regular_payload is None:
            regular_payload = _load_payload(Path(str(summary["regular_accepted_out_json"])))
        for label in candidate_labels:
            if label not in candidate_payloads:
                candidate_payloads[label] = _load_payload(Path(str(summary[f"{label}_accepted_out_json"])))

        if candidate_labels == ("frameselect",):
            comparison = compare_payloads(
                regular_payload,
                candidate_payloads["frameselect"],
                max_total_ratio=float(args.max_total_regression_ratio),
                max_backward_ratio=float(args.max_backward_regression_ratio),
                max_storage_ratio=float(args.max_storage_regression_ratio),
            )
        else:
            comparison = compare_candidate_set(
                regular_payload,
                candidate_payloads,
                max_total_ratio=float(args.max_total_regression_ratio),
                max_backward_ratio=float(args.max_backward_regression_ratio),
                max_storage_ratio=float(args.max_storage_regression_ratio),
            )
        summary["comparison"] = comparison
        summary["status"] = "ok" if comparison["status"] == "ok" else "comparison_failed"
        _write_summary(summary_json, summary)
        return 0 if comparison["status"] == "ok" else 2
    except (KeyboardInterrupt, _ComparisonInterrupted) as exc:
        previous_status = summary.get("status")
        summary["status"] = "interrupted"
        summary["interrupted_at"] = _iso_now()
        summary["interrupted_reason"] = str(exc) or type(exc).__name__
        summary["interrupted_previous_status"] = previous_status
        _write_summary(summary_json, summary)
        print(f"[factorized_frameselect_gate] interrupted: {summary['interrupted_reason']}", file=sys.stderr, flush=True)
        return 130
    finally:
        _restore_interrupt_handlers(previous_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
