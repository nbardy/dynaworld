from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import (
        ROOT,
        load_optional_report_json,
        mean_timing_without_first,
        run_star_uvt_feature_trainer_subprocess,
        write_report_csv,
        write_report_json,
        write_report_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import (
        ROOT,
        load_optional_report_json,
        mean_timing_without_first,
        run_star_uvt_feature_trainer_subprocess,
        write_report_csv,
        write_report_json,
        write_report_text,
    )
from config_utils import load_config_file


BASE_CONFIG = ROOT / "src/train_configs/star_uvt_feature_testvideo_8f_64_directatomic_20step.jsonc"
POLICIES = ("cadence", "measured")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != 0.0 and abs(value) < 1.0e-3:
            return f"{value:.6g}"
        return f"{value:.4f}"
    return str(value)


def _compatible_projective_config(
    base: dict[str, Any],
    *,
    policy: str,
    steps: int,
    refresh_every: int,
    support_guard_padding: float,
    support_guard_policy: str,
    support_guard_bisect_steps: int,
    support_stale_overshoot_epsilon: float,
    support_stale_tail_alpha_epsilon: float,
    tile_capacity: int | None,
    out_json: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["train"]["steps"] = int(steps)
    cfg["train"]["frame_chunk_size"] = None
    cfg["train"]["require_loss_decrease"] = False
    cfg["train"]["require_no_tile_overflow"] = False
    cfg["feature_uvt"]["feature_dim"] = 3
    cfg["feature_uvt"]["render_mode"] = "feature_direct_atomic"
    if tile_capacity is not None:
        cfg["feature_uvt"]["tile_capacity"] = int(tile_capacity)
    cfg["feature_uvt"]["projective_interval"] = {
        "enabled": True,
        "sigma_px": 2.0,
        "tile_size": 8,
        "uv_padding": 8.0,
        "support_guard_padding": float(support_guard_padding),
        "support_guard_policy": support_guard_policy,
        "support_guard_bisect_steps": int(support_guard_bisect_steps),
        "support_stale_overshoot_epsilon": float(support_stale_overshoot_epsilon),
        "support_stale_tail_alpha_epsilon": float(support_stale_tail_alpha_epsilon),
        "refresh_policy": policy,
        "refresh_every": int(refresh_every),
        "fallback_render_mode": "mixed",
    }
    cfg["colorize"]["hidden_dim"] = None
    cfg["output"]["out_json"] = str(out_json)
    cfg["output"]["contact_sheet"] = None
    cfg["output"]["side_by_side_video"] = None
    cfg["output"]["rgb_probe_contact_sheet"] = None
    cfg["output"]["rgb_probe_side_by_side_video"] = None
    cfg["logging"]["wandb_enabled"] = False
    cfg["logging"]["wandb_mode"] = "offline"
    cfg["logging"]["wandb_run_name"] = f"star-uvt-projective-interval-cache-{policy}"
    cfg["logging"]["wandb_tags"] = [
        "star_uvt",
        "projective_interval",
        "cache_policy",
        policy,
    ]
    return cfg


def _row_from_result(
    *,
    policy: str,
    support_guard_padding: float,
    support_guard_policy: str,
    support_guard_bisect_steps: int,
    support_stale_overshoot_epsilon: float,
    support_stale_tail_alpha_epsilon: float,
    tile_capacity: int | None,
    result: Any,
    payload: dict[str, Any] | None,
    out_json: Path,
    log_path: Path,
) -> dict[str, Any]:
    timing = payload.get("mean_timing_ms", {}) if isinstance(payload, dict) else {}
    last_timing = payload.get("last_timing_ms", {}) if isinstance(payload, dict) else {}
    tile_stats = payload.get("tile_stats", {}) if isinstance(payload, dict) else {}
    return {
        "policy": policy,
        "support_guard_padding": float(support_guard_padding),
        "support_guard_policy": support_guard_policy,
        "support_guard_bisect_steps": int(support_guard_bisect_steps),
        "support_stale_overshoot_epsilon": float(support_stale_overshoot_epsilon),
        "support_stale_tail_alpha_epsilon": float(support_stale_tail_alpha_epsilon),
        "tile_capacity": tile_capacity,
        "status": result.status,
        "error": result.error,
        "elapsed_sec": round(result.elapsed_sec, 3),
        "pass": payload.get("pass") if isinstance(payload, dict) else None,
        "steps": payload.get("steps") if isinstance(payload, dict) else None,
        "start_loss": payload.get("start_loss") if isinstance(payload, dict) else None,
        "end_loss": payload.get("end_loss") if isinstance(payload, dict) else None,
        "loss_decreased": payload.get("loss_decreased") if isinstance(payload, dict) else None,
        "mean_step_ms": timing.get("step_ms"),
        "no_first_step_ms": mean_timing_without_first(payload, "step_ms") if isinstance(payload, dict) else None,
        "last_step_ms": last_timing.get("step_ms") if isinstance(last_timing, dict) else None,
        "mean_render_forward_ms": timing.get("render_forward_ms"),
        "mean_backward_ms": timing.get("backward_ms"),
        "tile_overflow_sum": payload.get("tile_overflow_sum") if isinstance(payload, dict) else None,
        "max_tile_count": tile_stats.get("max_tile_count") if isinstance(tile_stats, dict) else None,
        "projective_interval_refresh_policy": (
            payload.get("projective_interval_refresh_policy") if isinstance(payload, dict) else None
        ),
        "projective_interval_refresh_every": (
            payload.get("projective_interval_refresh_every") if isinstance(payload, dict) else None
        ),
        "projective_interval_effective_support_uv_padding": (
            payload.get("projective_interval_effective_support_uv_padding") if isinstance(payload, dict) else None
        ),
        "projective_interval_cache_rebuilds": (
            payload.get("projective_interval_cache_rebuilds") if isinstance(payload, dict) else None
        ),
        "projective_interval_cache_live_updates": (
            payload.get("projective_interval_cache_live_updates") if isinstance(payload, dict) else None
        ),
        "projective_interval_cache_staleness_checks": (
            payload.get("projective_interval_cache_staleness_checks") if isinstance(payload, dict) else None
        ),
        "projective_interval_cache_stale_refreshes": (
            payload.get("projective_interval_cache_stale_refreshes") if isinstance(payload, dict) else None
        ),
        "projective_interval_cache_support_rebins": (
            payload.get("projective_interval_cache_support_rebins") if isinstance(payload, dict) else None
        ),
        "projective_interval_cache_last_support_missing_tile_pairs": (
            payload.get("projective_interval_cache_last_support_missing_tile_pairs") if isinstance(payload, dict) else None
        ),
        "projective_interval_cache_last_support_max_overshoot_px": (
            payload.get("projective_interval_cache_last_support_max_overshoot_px") if isinstance(payload, dict) else None
        ),
        "projective_interval_cache_max_support_max_overshoot_px": (
            payload.get("projective_interval_cache_max_support_max_overshoot_px") if isinstance(payload, dict) else None
        ),
        "projective_interval_cache_last_support_tail_alpha_bound": (
            payload.get("projective_interval_cache_last_support_tail_alpha_bound") if isinstance(payload, dict) else None
        ),
        "projective_interval_cache_max_support_tail_alpha_bound": (
            payload.get("projective_interval_cache_max_support_tail_alpha_bound") if isinstance(payload, dict) else None
        ),
        "projective_interval_cache_visibility_stratifications": (
            payload.get("projective_interval_cache_visibility_stratifications") if isinstance(payload, dict) else None
        ),
        "projective_interval_cache_fallback_marks": (
            payload.get("projective_interval_cache_fallback_marks") if isinstance(payload, dict) else None
        ),
        "json_path": str(out_json),
        "log_path": str(log_path),
    }


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = (
        "policy",
        "support_guard_padding",
        "support_guard_policy",
        "support_guard_bisect_steps",
        "support_stale_overshoot_epsilon",
        "support_stale_tail_alpha_epsilon",
        "tile_capacity",
        "status",
        "pass",
        "steps",
        "end_loss",
        "mean_step_ms",
        "no_first_step_ms",
        "last_step_ms",
        "mean_render_forward_ms",
        "mean_backward_ms",
        "tile_overflow_sum",
        "max_tile_count",
        "projective_interval_cache_rebuilds",
        "projective_interval_cache_live_updates",
        "projective_interval_cache_staleness_checks",
        "projective_interval_cache_stale_refreshes",
        "projective_interval_cache_support_rebins",
        "projective_interval_cache_last_support_missing_tile_pairs",
        "projective_interval_cache_last_support_max_overshoot_px",
        "projective_interval_cache_max_support_max_overshoot_px",
        "projective_interval_cache_last_support_tail_alpha_bound",
        "projective_interval_cache_max_support_tail_alpha_bound",
        "projective_interval_effective_support_uv_padding",
        "json_path",
    )
    lines = [
        "# STAR UVT Projective Interval Cache Policy Benchmark",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This benchmark runs the compatible `feature_dim=3`, full-frame projective",
        "interval route. It measures cache policy behavior, not the future F32",
        "anisotropic/pixel-depth endpoint.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key in columns) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `cadence` is the safe fixed full-atlas rebuild policy.",
            "- `measured` keeps the compiled atlas alive and checks support/visibility staleness before render.",
            "- `support_guard_padding` widens compiled support cells while correctness checks use base `uv_padding`.",
            "- The key contract is fewer full rebuilds with matching loss behavior.",
        ]
    )
    write_report_text(path, "\n".join(lines) + "\n")


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if not isinstance(value, int):
        errors.append(f"{label} must be an integer, got {value!r}")
        return 0
    return int(value)


def verify_projective_interval_cache_policy_report(payload: dict[str, object]) -> list[str]:
    """Return contract failures for a saved projective interval cache-policy report."""

    errors: list[str] = []
    steps = _finite_int(payload.get("steps"), "steps", errors)
    refresh_every = _finite_int(payload.get("refresh_every"), "refresh_every", errors)
    support_guard_padding = _finite_float(payload.get("support_guard_padding"), "support_guard_padding", errors)
    support_tail_epsilon = _finite_float(
        payload.get("support_stale_tail_alpha_epsilon"),
        "support_stale_tail_alpha_epsilon",
        errors,
    )
    support_overshoot_epsilon = _finite_float(
        payload.get("support_stale_overshoot_epsilon"),
        "support_stale_overshoot_epsilon",
        errors,
    )
    tile_capacity = payload.get("tile_capacity")
    if not isinstance(tile_capacity, int) or int(tile_capacity) <= 0:
        errors.append(f"tile_capacity must be a positive integer, got {tile_capacity!r}")
        tile_capacity_int = 0
    else:
        tile_capacity_int = int(tile_capacity)

    if steps <= 0:
        errors.append(f"steps must be positive, got {steps}")
    if refresh_every <= 0:
        errors.append(f"refresh_every must be positive, got {refresh_every}")
    if support_guard_padding <= 0.0:
        errors.append(f"support_guard_padding must be positive, got {support_guard_padding}")
    if payload.get("support_guard_policy") != "slack_budgeted":
        errors.append("support_guard_policy must be 'slack_budgeted' for certified aggregate reports")
    if support_overshoot_epsilon != 0.0:
        errors.append("support_stale_overshoot_epsilon must be 0.0 so reuse is tail-certified, not pixel-pardoned")
    if support_tail_epsilon <= 0.0:
        errors.append(f"support_stale_tail_alpha_epsilon must be positive, got {support_tail_epsilon}")

    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 2:
        errors.append("rows must contain exactly cadence and measured rows")
        return errors
    rows = [row for row in raw_rows if isinstance(row, dict)]
    if len(rows) != len(raw_rows):
        errors.append("all rows must be objects")
    by_policy = {str(row.get("policy")): row for row in rows}
    missing = sorted({"cadence", "measured"} - set(by_policy))
    if missing:
        errors.append(f"missing policy rows: {missing}")
        return errors

    for policy, row in by_policy.items():
        prefix = f"{policy} row"
        if row.get("status") != "ok":
            errors.append(f"{prefix} status must be ok")
        if row.get("pass") is not True:
            errors.append(f"{prefix} pass must be true")
        if row.get("loss_decreased") is not True:
            errors.append(f"{prefix} loss_decreased must be true")
        if row.get("projective_interval_refresh_policy") != policy:
            errors.append(f"{prefix} projective_interval_refresh_policy must match policy")
        if _finite_int(row.get("steps"), f"{prefix} steps", errors) != steps:
            errors.append(f"{prefix} steps must match report steps")
        if _finite_int(row.get("projective_interval_refresh_every"), f"{prefix} refresh_every", errors) != refresh_every:
            errors.append(f"{prefix} refresh_every must match report refresh_every")
        if _finite_int(row.get("tile_overflow_sum"), f"{prefix} tile_overflow_sum", errors) != 0:
            errors.append(f"{prefix} tile_overflow_sum must be zero")
        if _finite_int(row.get("max_tile_count"), f"{prefix} max_tile_count", errors) > tile_capacity_int:
            errors.append(f"{prefix} max_tile_count must fit tile_capacity")
        if _finite_int(row.get("projective_interval_cache_visibility_stratifications"), f"{prefix} visibility_stratifications", errors) != 0:
            errors.append(f"{prefix} visibility_stratifications must be zero for this support-only gate")
        if _finite_int(row.get("projective_interval_cache_fallback_marks"), f"{prefix} fallback_marks", errors) != 0:
            errors.append(f"{prefix} fallback_marks must be zero")
        if _finite_float(row.get("end_loss"), f"{prefix} end_loss", errors) > _finite_float(
            row.get("start_loss"),
            f"{prefix} start_loss",
            errors,
        ):
            errors.append(f"{prefix} end_loss must not exceed start_loss")
        if _finite_float(row.get("no_first_step_ms"), f"{prefix} no_first_step_ms", errors) <= 0.0:
            errors.append(f"{prefix} no_first_step_ms must be positive")
        last_tail = _finite_float(
            row.get("projective_interval_cache_last_support_tail_alpha_bound"),
            f"{prefix} last_support_tail_alpha_bound",
            errors,
        )
        max_tail = _finite_float(
            row.get("projective_interval_cache_max_support_tail_alpha_bound"),
            f"{prefix} max_support_tail_alpha_bound",
            errors,
        )
        if not 0.0 < last_tail <= support_tail_epsilon:
            errors.append(f"{prefix} last tail-alpha bound must be in (0, epsilon], got {last_tail}")
        if max_tail < last_tail:
            errors.append(f"{prefix} max tail-alpha bound must be >= last bound")

    cadence = by_policy["cadence"]
    measured = by_policy["measured"]
    cadence_rebuilds = _finite_int(cadence.get("projective_interval_cache_rebuilds"), "cadence rebuilds", errors)
    measured_rebuilds = _finite_int(measured.get("projective_interval_cache_rebuilds"), "measured rebuilds", errors)
    cadence_live = _finite_int(cadence.get("projective_interval_cache_live_updates"), "cadence live_updates", errors)
    measured_live = _finite_int(measured.get("projective_interval_cache_live_updates"), "measured live_updates", errors)
    cadence_rebins = _finite_int(
        cadence.get("projective_interval_cache_support_rebins"),
        "cadence support_rebins",
        errors,
    )
    measured_rebins = _finite_int(
        measured.get("projective_interval_cache_support_rebins"),
        "measured support_rebins",
        errors,
    )
    measured_stale_refreshes = _finite_int(
        measured.get("projective_interval_cache_stale_refreshes"),
        "measured stale_refreshes",
        errors,
    )
    measured_staleness_checks = _finite_int(
        measured.get("projective_interval_cache_staleness_checks"),
        "measured staleness_checks",
        errors,
    )
    measured_max_tail = _finite_float(
        measured.get("projective_interval_cache_max_support_tail_alpha_bound"),
        "measured max_support_tail_alpha_bound",
        errors,
    )

    if cadence_rebuilds < 2:
        errors.append("cadence must perform repeated rebuilds")
    if measured_rebuilds >= cadence_rebuilds:
        errors.append("measured policy must use fewer full rebuilds than cadence")
    if measured_live <= cadence_live:
        errors.append("measured policy must use more live atlas updates than cadence")
    if measured_staleness_checks < measured_live:
        errors.append("measured staleness checks must cover live updates")
    if cadence_rebins != 0:
        errors.append("cadence support_rebins must be zero")
    if measured_rebins != measured_stale_refreshes:
        errors.append("measured support_rebins must equal stale_refreshes")
    if measured_rebins > cadence_rebuilds:
        errors.append("measured support_rebins must stay below cadence rebuild count")
    if measured_rebins == 0 and measured_max_tail > support_tail_epsilon:
        errors.append("measured zero-rebin case must keep max tail-alpha bound within epsilon")
    if measured_rebins > 0 and measured_max_tail <= support_tail_epsilon:
        errors.append("measured rebin case must show a max tail-alpha bound above epsilon")

    cadence_loss = _finite_float(cadence.get("end_loss"), "cadence end_loss", errors)
    measured_loss = _finite_float(measured.get("end_loss"), "measured end_loss", errors)
    if abs(measured_loss - cadence_loss) > 1.0e-9:
        errors.append(f"measured end_loss must match cadence end_loss, got delta {measured_loss - cadence_loss}")
    if _finite_float(measured.get("no_first_step_ms"), "measured no_first_step_ms", errors) >= _finite_float(
        cadence.get("no_first_step_ms"),
        "cadence no_first_step_ms",
        errors,
    ):
        errors.append("measured no-first-step timing must be lower than cadence")

    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        errors.append("comparison must be present")
        return errors
    expected_rebuild_delta = measured_rebuilds - cadence_rebuilds
    expected_live_delta = measured_live - cadence_live
    expected_rebin_delta = measured_rebins - cadence_rebins
    if _finite_int(comparison.get("rebuild_delta_measured_minus_cadence"), "comparison rebuild_delta", errors) != expected_rebuild_delta:
        errors.append("comparison rebuild_delta does not match rows")
    if _finite_int(comparison.get("live_update_delta_measured_minus_cadence"), "comparison live_update_delta", errors) != expected_live_delta:
        errors.append("comparison live_update_delta does not match rows")
    if _finite_int(comparison.get("support_rebin_delta_measured_minus_cadence"), "comparison support_rebin_delta", errors) != expected_rebin_delta:
        errors.append("comparison support_rebin_delta does not match rows")
    loss_delta = _finite_float(comparison.get("end_loss_delta_measured_minus_cadence"), "comparison end_loss_delta", errors)
    if abs(loss_delta - (measured_loss - cadence_loss)) > 1.0e-9:
        errors.append("comparison end_loss_delta does not match rows")
    timing_delta = _finite_float(
        comparison.get("no_first_step_ms_delta_measured_minus_cadence"),
        "comparison no_first_step_ms_delta",
        errors,
    )
    if timing_delta >= 0.0:
        errors.append("comparison no_first_step_ms_delta must be negative")

    return errors


def assert_projective_interval_cache_policy_report(payload: dict[str, object]) -> None:
    errors = verify_projective_interval_cache_policy_report(payload)
    if errors:
        raise AssertionError("projective interval cache-policy report failed:\n- " + "\n- ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--refresh-every", type=int, default=2)
    parser.add_argument("--support-guard-padding", type=float, default=0.0)
    parser.add_argument(
        "--support-guard-policy",
        choices=("fixed", "budgeted", "local_budgeted", "trace_budgeted", "slack_budgeted"),
        default="fixed",
    )
    parser.add_argument("--support-guard-bisect-steps", type=int, default=8)
    parser.add_argument("--support-stale-overshoot-epsilon", type=float, default=0.0)
    parser.add_argument("--support-stale-tail-alpha-epsilon", type=float, default=0.0)
    parser.add_argument("--tile-capacity", type=int)
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-report", type=Path, help="validate an existing summary.json without rerunning")
    args = parser.parse_args()

    if args.verify_report is not None:
        payload = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_projective_interval_cache_policy_report(payload)
        print(f"verified {args.verify_report}")
        return

    base = load_config_file(args.base_config)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.out_dir or ROOT / "outputs/benchmarks" / f"{timestamp}_star_uvt_projective_interval_cache_policy"
    config_dir = out_dir / "configs"
    log_dir = out_dir / "logs"
    case_dir = out_dir / "cases"
    for directory in (config_dir, log_dir, case_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    configs: dict[str, str] = {}
    for policy in POLICIES:
        out_json = case_dir / f"{policy}.json"
        config_path = config_dir / f"{policy}.json"
        log_path = log_dir / f"{policy}.log"
        cfg = _compatible_projective_config(
            base,
            policy=policy,
            steps=args.steps,
            refresh_every=args.refresh_every,
            support_guard_padding=args.support_guard_padding,
            support_guard_policy=args.support_guard_policy,
            support_guard_bisect_steps=args.support_guard_bisect_steps,
            support_stale_overshoot_epsilon=args.support_stale_overshoot_epsilon,
            support_stale_tail_alpha_epsilon=args.support_stale_tail_alpha_epsilon,
            tile_capacity=args.tile_capacity,
            out_json=out_json,
        )
        write_report_json(config_path, cfg)
        configs[policy] = str(config_path)
        if args.dry_run:
            rows.append(
                {
                    "policy": policy,
                    "support_guard_padding": float(args.support_guard_padding),
                    "support_guard_policy": args.support_guard_policy,
                    "support_guard_bisect_steps": int(args.support_guard_bisect_steps),
                    "support_stale_overshoot_epsilon": float(args.support_stale_overshoot_epsilon),
                    "support_stale_tail_alpha_epsilon": float(args.support_stale_tail_alpha_epsilon),
                    "tile_capacity": args.tile_capacity,
                    "status": "dry_run",
                    "json_path": str(out_json),
                    "log_path": str(log_path),
                }
            )
            continue
        if out_json.exists():
            out_json.unlink()
        result = run_star_uvt_feature_trainer_subprocess(
            config_path=config_path,
            log_path=log_path,
            python=args.python,
            timeout_sec=args.timeout_sec,
            tmp_dir=out_dir / "tmp",
            env_overrides=(
                None if args.tile_capacity is None else {"STAR_UVT_TILE_CAPACITY": int(args.tile_capacity)}
            ),
        )
        payload = load_optional_report_json(out_json)
        rows.append(
            _row_from_result(
                policy=policy,
                support_guard_padding=args.support_guard_padding,
                support_guard_policy=args.support_guard_policy,
                support_guard_bisect_steps=args.support_guard_bisect_steps,
                support_stale_overshoot_epsilon=args.support_stale_overshoot_epsilon,
                support_stale_tail_alpha_epsilon=args.support_stale_tail_alpha_epsilon,
                tile_capacity=args.tile_capacity,
                result=result,
                payload=payload,
                out_json=out_json,
                log_path=log_path,
            )
        )

    summary: dict[str, Any] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "base_config": str(args.base_config),
        "steps": int(args.steps),
        "refresh_every": int(args.refresh_every),
        "support_guard_padding": float(args.support_guard_padding),
        "support_guard_policy": args.support_guard_policy,
        "support_guard_bisect_steps": int(args.support_guard_bisect_steps),
        "support_stale_overshoot_epsilon": float(args.support_stale_overshoot_epsilon),
        "support_stale_tail_alpha_epsilon": float(args.support_stale_tail_alpha_epsilon),
        "tile_capacity": args.tile_capacity,
        "configs": configs,
        "rows": rows,
    }
    if len(rows) == 2 and all(row.get("status") == "ok" for row in rows):
        by_policy = {row["policy"]: row for row in rows}
        cadence = by_policy["cadence"]
        measured = by_policy["measured"]
        summary["comparison"] = {
            "rebuild_delta_measured_minus_cadence": (
                int(measured["projective_interval_cache_rebuilds"])
                - int(cadence["projective_interval_cache_rebuilds"])
            ),
            "live_update_delta_measured_minus_cadence": (
                int(measured["projective_interval_cache_live_updates"])
                - int(cadence["projective_interval_cache_live_updates"])
            ),
            "end_loss_delta_measured_minus_cadence": float(measured["end_loss"]) - float(cadence["end_loss"]),
            "support_rebin_delta_measured_minus_cadence": (
                int(measured["projective_interval_cache_support_rebins"])
                - int(cadence["projective_interval_cache_support_rebins"])
            ),
            "no_first_step_ms_delta_measured_minus_cadence": (
                None
                if measured["no_first_step_ms"] is None or cadence["no_first_step_ms"] is None
                else float(measured["no_first_step_ms"]) - float(cadence["no_first_step_ms"])
            ),
        }

    write_report_json(out_dir / "summary.json", summary)
    write_report_csv(out_dir / "rows.csv", rows)
    _write_markdown(rows, out_dir / "summary.md")
    print(out_dir)


if __name__ == "__main__":
    main()
