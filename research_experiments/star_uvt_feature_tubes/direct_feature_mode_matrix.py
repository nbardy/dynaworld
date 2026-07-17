from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import (
        ROOT,
        STAR_UVT_ROOT,
        TRAIN_ROOT,
        load_optional_report_json,
        run_logged_subprocess,
        split_csv_ints,
        split_csv_strings,
        write_report_csv,
        write_report_json,
        write_report_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import (
        ROOT,
        STAR_UVT_ROOT,
        TRAIN_ROOT,
        load_optional_report_json,
        run_logged_subprocess,
        split_csv_ints,
        split_csv_strings,
        write_report_csv,
        write_report_json,
        write_report_text,
    )


BENCHMARK = ROOT / "research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py"

DEFAULT_MODES = (
    "direct_atomic",
    "direct_atomic_cached_bins",
    "gradcache",
    "gradcache_cached_bins",
    "gradcache_skip_feature_grad",
    "direct_atomic_feature_grad_only",
    "gradcache_feature_grad_only",
    "gradcache_feature_grad_only_reduce",
    "gradcache_feature_grad_only_reduce_vec4",
    "gradcache_two_pass_feature_grad",
    "gradcache_two_pass_feature_grad_reduce",
    "gradcache_two_pass_feature_grad_reduce_vec4",
    "gradcache_reduce_feature_grad",
    "gradcache_reduce_feature_grad_cached_bins",
    "gradcache_reduce_feature_grad_vec4",
    "gradcache_reduce_feature_grad_vec4_cached_bins",
    "fused_first3_sigmoid_mse",
    "linear_sigmoid_mse",
    "linear_sigmoid_mse_skip_colorizer_grad",
    "logit_handoff",
    "logit_handoff_reduce",
    "logit_handoff_reduce_vec4",
)


@dataclass(frozen=True)
class MatrixCase:
    label: str
    mode: str
    size: int
    out_json: Path
    command: tuple[str, ...]


def _run_case(case: MatrixCase, *, log_dir: Path, timeout_sec: int) -> dict[str, Any]:
    log_path = log_dir / f"{case.label}.log"
    if case.out_json.exists():
        case.out_json.unlink()
    result = run_logged_subprocess(
        case.command,
        log_path=log_path,
        cwd=ROOT,
        timeout_sec=timeout_sec,
        pythonpath=(TRAIN_ROOT, STAR_UVT_ROOT),
        tmp_dir=log_dir.parent / "tmp",
    )
    payload = load_optional_report_json(case.out_json)
    timing = payload.get("timing", {}) if isinstance(payload, dict) else {}
    return {
        "case": case.label,
        "mode": case.mode,
        "status": result.status,
        "error": result.error,
        "elapsed_sec": round(result.elapsed_sec, 3),
        "pass": payload.get("pass") if isinstance(payload, dict) else None,
        "frames": timing.get("frames"),
        "size": timing.get("size", case.size),
        "tubes": timing.get("tubes"),
        "feature_dim": timing.get("feature_dim"),
        "kernel_backward_mode": timing.get("kernel_backward_mode"),
        "cached_bins": timing.get("cached_bins"),
        "two_pass_feature_grad": timing.get("two_pass_feature_grad"),
        "two_pass_feature_mode": timing.get("two_pass_feature_mode"),
        "feature_grad_only": timing.get("feature_grad_only"),
        "forward_ms": timing.get("forward_ms"),
        "handoff_prep_ms": timing.get("handoff_prep_ms"),
        "backward_ms": timing.get("backward_ms"),
        "total_ms": timing.get("total_ms"),
        "tile_overflow_sum": timing.get("tile_overflow_sum"),
        "tile_unstable_sum": timing.get("tile_unstable_sum"),
        "fixedbin_eligible": (
            None
            if timing.get("tile_overflow_sum") is None
            else int(timing.get("tile_overflow_sum", 0)) == 0
        ),
        "feature_grad_skipped": timing.get("feature_grad_skipped"),
        "colorizer_grad_skipped": timing.get("colorizer_grad_skipped"),
        "json_path": str(case.out_json),
        "log_path": str(log_path),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "mode",
        "status",
        "pass",
        "size",
        "frames",
        "tubes",
        "feature_dim",
        "kernel_backward_mode",
        "cached_bins",
        "two_pass_feature_grad",
        "two_pass_feature_mode",
        "feature_grad_only",
        "total_ms",
        "forward_ms",
        "handoff_prep_ms",
        "backward_ms",
        "tile_overflow_sum",
        "fixedbin_eligible",
        "json_path",
    )
    lines = [
        "# STAR UVT Feature Direct Mode Matrix",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key in columns) + " |")
    write_report_text(path, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--sizes", default="128,256")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--tubes", type=int, default=32768)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--feature-dims", default="4,32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def main() -> None:
    args = parse_args()
    modes = split_csv_strings(args.modes)
    sizes = split_csv_ints(args.sizes)
    unknown = [mode for mode in modes if mode not in DEFAULT_MODES]
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.out_dir or ROOT / "outputs/benchmarks" / f"{timestamp}_star_uvt_feature_direct_mode_matrix"
    cases_dir = out_dir / "cases"
    log_dir = out_dir / "logs"
    cases_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cases: list[MatrixCase] = []
    for size in sizes:
        for mode in modes:
            label = f"feature_direct_{args.frames}f_{size}px_{args.tubes}t_F{args.feature_dim}_{mode}"
            out_json = cases_dir / f"{label}.json"
            command = (
                args.python,
                str(BENCHMARK),
                "--feature-dims",
                args.feature_dims,
                "--timing-frames",
                str(args.frames),
                "--timing-size",
                str(size),
                "--timing-tubes",
                str(args.tubes),
                "--timing-feature-dim",
                str(args.feature_dim),
                "--backward-mode",
                mode,
                "--timing-warmup",
                str(args.warmup),
                "--timing-repeat",
                str(args.repeat),
                "--out-json",
                str(out_json),
            )
            cases.append(MatrixCase(label=label, mode=mode, size=size, out_json=out_json, command=command))

    manifest = {
        "cases": [case.label for case in cases],
        "args": _jsonable_args(args),
        "notes": [
            "Run modes sequentially; parallel MPS runs create invalid timings.",
            "Diagnostic modes with skipped gradients or fused losses are not trainer defaults.",
            "Cached-bin modes reuse forward bins for backward and report their effective kernel_backward_mode.",
            "Each subprocess uses an artifact-local TMPDIR and deletes stale case JSON before launch.",
        ],
    }
    write_report_json(out_dir / "manifest.json", manifest)

    if args.dry_run:
        for case in cases:
            print(case.label + ": " + " ".join(case.command))
        return

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.label}", flush=True)
        rows.append(_run_case(case, log_dir=log_dir, timeout_sec=args.timeout_sec))
    write_report_csv(out_dir / "summary.csv", rows)
    _write_markdown(rows, out_dir / "summary.md")
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
