from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .star_uvt_feature_tubes.report_artifacts import (
        ROOT,
        load_optional_report_json,
        run_logged_subprocess,
        write_report_csv,
        write_report_json,
        write_report_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from star_uvt_feature_tubes.report_artifacts import (
        ROOT,
        load_optional_report_json,
        run_logged_subprocess,
        write_report_csv,
        write_report_json,
        write_report_text,
    )

V0_ROOT = ROOT / "third_party/fast-mac-gsplat/variants/star_uvt_v0"
PRT_ROOT = ROOT / "third_party/fast-mac-gsplat/variants/star_uvt_prt_v0"


V0_SAMPLE_MODE_CASES: tuple[tuple[str, str], ...] = (
    ("direct_atomic", "index_add"),
    ("direct_fixedpoint", "index_add"),
    ("direct_split_fixedpoint", "index_add"),
    ("direct_serial", "index_add"),
    ("tile_pair_atomic", "index_add"),
    ("tile_pair_fixedpoint", "index_add"),
    ("tile_pair_reduced", "index_add"),
    ("tile_pair_reduced_parallel", "index_add"),
    ("tile_pair_suffix_reduced", "index_add"),
    ("atomic_append", "index_add"),
    ("with_keys", "key_sort_segmented_metal"),
    ("tile_pair", "key_sort_segmented_metal"),
    ("tile_pair_compensated", "key_sort_segmented_metal"),
    ("tile_pair_grouped", "key_sort_segmented_metal"),
    ("tile_pair_parallel", "key_sort_segmented_metal"),
    ("tile_pair_scanline", "key_sort_segmented_metal"),
    ("tile_pair_sharedsort", "key_sort_segmented_metal"),
    ("tile_pair_target_bounds", "key_sort_segmented_metal"),
    ("tile_pair_suffix", "key_sort_segmented_metal"),
)

PRT_BACKWARD_MODES: tuple[str, ...] = (
    "direct_serial",
    "tile_pair_atomic",
    "tile_pixel_atomic",
)


@dataclass(frozen=True)
class KernelCase:
    variant: str
    benchmark: str
    label: str
    command: tuple[str, ...]
    env_path: Path
    out_json: Path
    timeout_sec: int


def _summary_value(value: Any, key: str = "median") -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _base_row(case: KernelCase, status: str, elapsed_sec: float, error: str = "") -> dict[str, Any]:
    return {
        "variant": case.variant,
        "benchmark": case.benchmark,
        "case": case.label,
        "status": status,
        "elapsed_sec": round(elapsed_sec, 3),
        "error": error,
    }


def _summarize_v0(case: KernelCase, payload: dict[str, Any] | None, status: str, elapsed_sec: float, error: str) -> dict[str, Any]:
    row = _base_row(case, status, elapsed_sec, error)
    if not payload:
        return row
    row.update(
        {
            "target_size": payload.get("target_size"),
            "frames": payload.get("frames"),
            "tube_count": payload.get("tube_count"),
            "mode": payload.get("sample_emission_mode"),
            "reducer": payload.get("reduction_mode"),
            "sample_unit": payload.get("sample_unit"),
            "sample_count": payload.get("sample_count"),
            "valid_sample_count": payload.get("valid_sample_count"),
            "allocated_sample_slot_count": payload.get("allocated_sample_slot_count"),
            "compact_sample_fraction": payload.get("compact_sample_fraction"),
            "sample_backward_ms": _summary_value(payload.get("sample_backward_ms")),
            "reduce_bundle_ms": _summary_value(payload.get("reduce_bundle_ms")),
            "sample_plus_reduce_ms": payload.get("sample_plus_reduce_median_ms"),
            "unstable_tile_fraction": payload.get("unstable_tile_fraction"),
        }
    )
    return row


def _summarize_prt_train(
    case: KernelCase,
    payload: dict[str, Any] | None,
    status: str,
    elapsed_sec: float,
    error: str,
) -> dict[str, Any]:
    row = _base_row(case, status, elapsed_sec, error)
    if not payload:
        return row
    rows = payload.get("rows") or []
    first = rows[0] if rows else {}
    row.update(
        {
            "target_size": first.get("width"),
            "frames": first.get("frames"),
            "tube_count": first.get("tube_count"),
            "mode": first.get("backward_mode"),
            "reducer": payload.get("tile_config_key"),
            "sample_unit": "projective_rational_train_step",
            "forward_ms": first.get("median_forward_ms"),
            "loss_ms": first.get("median_loss_ms"),
            "backward_ms": first.get("median_backward_ms"),
            "optimizer_ms": first.get("median_optimizer_ms"),
            "wall_ms": first.get("median_wall_ms"),
            "tile_pair_count": first.get("tile_pair_count"),
            "overflow_tile_count": first.get("overflow_tile_count"),
            "pass": first.get("pass"),
        }
    )
    return row


def _summarize_prt_fused(
    case: KernelCase,
    payload: dict[str, Any] | None,
    status: str,
    elapsed_sec: float,
    error: str,
) -> dict[str, Any]:
    row = _base_row(case, status, elapsed_sec, error)
    if not payload:
        return row
    rows = payload.get("rows") or []
    first = rows[0] if rows else {}
    row.update(
        {
            "target_size": first.get("width"),
            "frames": first.get("frames"),
            "tube_count": first.get("tube_count"),
            "mode": "tile_pixel_fused_mse",
            "reducer": payload.get("tile_config_key"),
            "sample_unit": "projective_rational_fused_mse",
            "separate_median_ms": first.get("separate_median_ms"),
            "fused_median_ms": first.get("fused_median_ms"),
            "fused_speedup": first.get("fused_speedup"),
            "max_grad_abs_error": first.get("max_grad_abs_error"),
            "max_grad_rel_error": first.get("max_grad_rel_error"),
            "fused_overflow_tile_count": first.get("fused_overflow_tile_count"),
            "pass": first.get("pass"),
        }
    )
    return row


def _run_case(case: KernelCase, log_dir: Path) -> tuple[dict[str, Any], Path]:
    log_path = log_dir / f"{case.label}.log"
    completed = run_logged_subprocess(
        case.command,
        log_path=log_path,
        cwd=ROOT,
        timeout_sec=case.timeout_sec,
        pythonpath=(case.env_path,),
    )
    status = completed.status
    error = completed.error
    elapsed_sec = completed.elapsed_sec
    payload = load_optional_report_json(case.out_json)
    if status == "failed" and payload is not None:
        status = "completed_nonpassing"
    if case.benchmark == "v0_backward_breakdown":
        row = _summarize_v0(case, payload, status, elapsed_sec, error)
    elif case.benchmark == "prt_train_step_breakdown":
        row = _summarize_prt_train(case, payload, status, elapsed_sec, error)
    elif case.benchmark == "prt_fused_mse_timing":
        row = _summarize_prt_fused(case, payload, status, elapsed_sec, error)
    else:
        row = _base_row(case, status, elapsed_sec, error)
    return row, log_path


def _v0_cases(args: argparse.Namespace, out_dir: Path, python: str) -> list[KernelCase]:
    cases = []
    probe = V0_ROOT / "research_project/benchmarks/uvt_backward_breakdown_probe.py"
    requested = set(args.v0_mode_cases) if args.v0_mode_cases else None
    for emission_mode, reduction_mode in V0_SAMPLE_MODE_CASES:
        case_key = f"{emission_mode}+{reduction_mode}"
        if requested is not None and emission_mode not in requested and case_key not in requested:
            continue
        label = f"v0_{args.target_size}_{args.frames}f_{args.tube_count}_{emission_mode}_{reduction_mode}"
        out_json = out_dir / "cases" / f"{label}.json"
        command = (
            python,
            str(probe),
            args.video_path,
            "--target-size",
            str(args.target_size),
            "--max-frames",
            str(args.frames),
            "--tube-count",
            str(args.tube_count),
            "--seed",
            str(args.seed),
            "--spatial-precision",
            str(args.spatial_precision),
            "--temporal-precision",
            str(args.temporal_precision),
            "--opacity",
            str(args.opacity),
            "--uvt-tile-t",
            str(args.tile_t),
            "--uvt-tile-capacity",
            str(args.tile_capacity),
            "--uvt-sample-emission-mode",
            emission_mode,
            "--uvt-reduction-mode",
            reduction_mode,
            "--warmup-iterations",
            str(args.warmups),
            "--iterations",
            str(args.repeats),
            "--out-json",
            str(out_json),
        )
        cases.append(
            KernelCase(
                variant="star_uvt_v0",
                benchmark="v0_backward_breakdown",
                label=label,
                command=command,
                env_path=V0_ROOT,
                out_json=out_json,
                timeout_sec=args.timeout_sec,
            )
        )
    return cases


def _prt_cases(args: argparse.Namespace, out_dir: Path, python: str) -> list[KernelCase]:
    cases: list[KernelCase] = []
    train_probe = PRT_ROOT / "research_project/benchmarks/projective_rational_train_step_breakdown_probe.py"
    fused_probe = PRT_ROOT / "research_project/benchmarks/projective_rational_fused_mse_timing_probe.py"
    for mode in PRT_BACKWARD_MODES:
        label = f"prt_train_{args.prt_width}x{args.prt_height}_{args.prt_frames}f_{args.prt_tube_count}_{mode}"
        out_json = out_dir / "cases" / f"{label}.json"
        command = (
            python,
            str(train_probe),
            "--tube-counts",
            str(args.prt_tube_count),
            "--frames",
            str(args.prt_frames),
            "--width",
            str(args.prt_width),
            "--height",
            str(args.prt_height),
            "--tile-config",
            args.prt_tile_config,
            "--warmups",
            str(args.prt_warmups),
            "--repeats",
            str(args.prt_repeats),
            "--seed",
            str(args.seed),
            "--camera-motion-scale",
            str(args.prt_camera_motion_scale),
            "--forward-mode",
            "tiled",
            "--backward-mode",
            mode,
            "--out-json",
            str(out_json),
        )
        cases.append(
            KernelCase(
                variant="star_uvt_prt_v0",
                benchmark="prt_train_step_breakdown",
                label=label,
                command=command,
                env_path=PRT_ROOT,
                out_json=out_json,
                timeout_sec=args.prt_timeout_sec,
            )
        )
    label = f"prt_fused_mse_{args.prt_width}x{args.prt_height}_{args.prt_frames}f_{args.prt_tube_count}"
    out_json = out_dir / "cases" / f"{label}.json"
    command = (
        python,
        str(fused_probe),
        "--tube-counts",
        str(args.prt_tube_count),
        "--frames",
        str(args.prt_frames),
        "--width",
        str(args.prt_width),
        "--height",
        str(args.prt_height),
        "--warmups",
        str(args.prt_warmups),
        "--repeats",
        str(args.prt_repeats),
        "--seed",
        str(args.seed),
        "--camera-motion-scale",
        str(args.prt_camera_motion_scale),
        "--out-json",
        str(out_json),
    )
    if args.prt_tile_config == "auto":
        command += ("--prt-tile-policy", "train_speed")
    else:
        command += ("--tile-config", args.prt_tile_config)
    cases.append(
        KernelCase(
            variant="star_uvt_prt_v0",
            benchmark="prt_fused_mse_timing",
            label=label,
            command=command,
            env_path=PRT_ROOT,
            out_json=out_json,
            timeout_sec=args.prt_timeout_sec,
        )
    )
    return cases


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    sortable = sorted(
        rows,
        key=lambda row: (
            row.get("status") != "ok",
            float(row.get("sample_plus_reduce_ms") or row.get("backward_ms") or row.get("fused_median_ms") or 1.0e12),
        ),
    )
    lines = [
        "# STAR UVT Backward Kernel Matrix",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| variant | case | status | primary ms | detail |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in sortable:
        primary = row.get("sample_plus_reduce_ms") or row.get("backward_ms") or row.get("fused_median_ms")
        if isinstance(primary, float):
            primary_text = f"{primary:.3f}"
        else:
            primary_text = "" if primary is None else str(primary)
        detail_bits = []
        for key in (
            "mode",
            "reducer",
            "sample_unit",
            "sample_count",
            "compact_sample_fraction",
            "sample_backward_ms",
            "reduce_bundle_ms",
            "forward_ms",
            "loss_ms",
            "wall_ms",
            "separate_median_ms",
            "fused_speedup",
            "error",
        ):
            value = row.get(key)
            if value not in (None, ""):
                if isinstance(value, float):
                    value = f"{value:.3f}"
                detail_bits.append(f"{key}={value}")
        lines.append(
            f"| {row.get('variant', '')} | {row.get('case', '')} | {row.get('status', '')} | {primary_text} | "
            + ", ".join(detail_bits)
            + " |"
        )
    write_report_text(path, "\n".join(lines) + "\n")


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-path",
        default="data/youtube_curated_spans/raw/hlaZbH_OFBU_seg_003_s00131000_e00138000.mp4",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--include-v0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-prt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--tube-count", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--spatial-precision", type=float, default=0.125)
    parser.add_argument("--temporal-precision", type=float, default=2.0)
    parser.add_argument("--opacity", type=float, default=0.7)
    parser.add_argument("--tile-t", type=int, default=1)
    parser.add_argument("--tile-capacity", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument(
        "--v0-mode-cases",
        default="",
        help=(
            "Comma-separated screen-space STAR UVT modes to include. "
            "Accepts either emission mode names such as direct_atomic or "
            "emission+reducer keys such as direct_atomic+index_add. "
            "Empty means all known v0 modes."
        ),
    )

    parser.add_argument("--prt-width", type=int, default=256)
    parser.add_argument("--prt-height", type=int, default=256)
    parser.add_argument("--prt-frames", type=int, default=16)
    parser.add_argument("--prt-tube-count", type=int, default=512)
    parser.add_argument("--prt-tile-config", default="auto")
    parser.add_argument("--prt-camera-motion-scale", type=float, default=1.0)
    parser.add_argument("--prt-warmups", type=int, default=0)
    parser.add_argument("--prt-repeats", type=int, default=1)
    parser.add_argument("--prt-timeout-sec", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.v0_mode_cases = tuple(item.strip() for item in str(args.v0_mode_cases).split(",") if item.strip())
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.out_dir or ROOT / "outputs/benchmarks" / f"{timestamp}_star_uvt_backward_kernel_matrix"
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    python = str((ROOT / args.python).absolute()) if not Path(args.python).is_absolute() else args.python
    cases: list[KernelCase] = []
    if args.include_v0:
        cases.extend(_v0_cases(args, out_dir, python))
    if args.include_prt:
        cases.extend(_prt_cases(args, out_dir, python))

    manifest = {
        "out_dir": str(out_dir),
        "case_count": len(cases),
        "include_v0": args.include_v0,
        "include_prt": args.include_prt,
        "args": _jsonable_args(args),
        "notes": [
            "star_uvt_v0 screen-space cases are comparable within the same target/frame/tube setting.",
            "star_uvt_prt_v0 projective-rational cases use their own tube policy and are not apples-to-apples with 32768 screen UVT tubes.",
            "star_prt_v0 has a scaffold compact backward that raises at runtime; spacetime_v0 is not buildable yet, so neither is included as a runnable timing row.",
        ],
    }
    write_report_json(out_dir / "manifest.json", manifest)

    if args.dry_run:
        for case in cases:
            print(case.label + ": " + " ".join(case.command))
        return

    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.label}", flush=True)
        row, log_path = _run_case(case, log_dir)
        row["log_path"] = str(log_path)
        row["json_path"] = str(case.out_json)
        rows.append(row)
        print(f"  status={row['status']}", flush=True)

    write_report_csv(out_dir / "summary.csv", rows)
    _write_markdown(rows, out_dir / "summary.md")
    print(f"summary_csv={out_dir / 'summary.csv'}")
    print(f"summary_md={out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
