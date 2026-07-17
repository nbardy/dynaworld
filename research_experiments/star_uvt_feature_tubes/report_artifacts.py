from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "src" / "train"
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
TRAIN_DISPATCHER = ROOT / "src" / "train" / "train.py"
STAR_UVT_FEATURE_TRAINER = TRAIN_DISPATCHER
for path in (ROOT, TRAIN_ROOT, STAR_UVT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_artifacts import write_csv, write_json, write_text  # noqa: E402


@dataclass(frozen=True)
class LoggedSubprocessResult:
    status: str
    error: str
    elapsed_sec: float
    command: tuple[str, ...]


def root_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def load_report_json(path: str | Path) -> dict[str, Any]:
    with root_path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def load_optional_report_json(path: str | Path) -> dict[str, Any] | None:
    resolved = root_path(path)
    return load_report_json(resolved) if resolved.exists() else None


def load_optional_report_json_or_error(path: str | Path) -> dict[str, Any] | None:
    resolved = root_path(path)
    if not resolved.exists():
        return None
    try:
        return load_report_json(resolved)
    except Exception as exc:  # pragma: no cover - report builders preserve diagnostics
        return {"_load_error": str(exc)}


def split_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def split_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in split_csv_strings(value))


def split_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in split_csv_strings(value))


def fmt_cell(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pair(values: list[Any], digits: int = 3) -> str:
    return f"{fmt_cell(values[0], digits)} -> {fmt_cell(values[1], digits)}"


def write_report_json(path: str | Path, payload: Any) -> Path:
    return write_json(root_path(path), payload)


def write_report_text(path: str | Path, text: str) -> Path:
    return write_text(root_path(path), text)


def write_report_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> Path:
    row_list = list(rows)
    if fieldnames is None:
        ordered_fieldnames: list[str] = []
        for row in row_list:
            for key in row:
                if key not in ordered_fieldnames:
                    ordered_fieldnames.append(str(key))
        fieldnames = ordered_fieldnames
    return write_csv(root_path(path), row_list, fieldnames=fieldnames)


def read_report_csv(path: str | Path) -> list[dict[str, str]]:
    with root_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_timing_without_first(row: Mapping[str, Any], key: str) -> float | None:
    timings = row.get("step_timings_ms")
    if not isinstance(timings, list) or len(timings) <= 1:
        return None
    values = [float(item[key]) for item in timings[1:] if isinstance(item, Mapping) and key in item]
    return None if not values else sum(values) / float(len(values))


def summary_stats(values: Iterable[float]) -> dict[str, Any]:
    samples = list(values)
    ordered = sorted(samples)
    if not ordered:
        return {"samples": [], "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "samples": samples,
        "mean": sum(samples) / float(len(samples)),
        "min": ordered[0],
        "max": ordered[-1],
    }


def distribution_stats(values: Iterable[float]) -> dict[str, Any]:
    samples = list(values)
    if not samples:
        return {"count": 0, "mean": None, "min": None, "max": None, "stdev": None, "samples": []}
    return {
        "count": len(samples),
        "mean": statistics.fmean(samples),
        "min": min(samples),
        "max": max(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples": samples,
    }


def run_logged_subprocess(
    command: Sequence[str | Path],
    *,
    log_path: str | Path,
    cwd: str | Path = ROOT,
    timeout_sec: int | None = None,
    pythonpath: Iterable[str | Path] = (),
    env_defaults: Mapping[str, str | Path | int | float] | None = None,
    env_overrides: Mapping[str, str | Path | int | float] | None = None,
    tmp_dir: str | Path | None = None,
) -> LoggedSubprocessResult:
    resolved_log_path = root_path(log_path)
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    pythonpath_values = [str(root_path(path)) for path in pythonpath]
    if pythonpath_values:
        env["PYTHONPATH"] = ":".join(pythonpath_values)
    if tmp_dir is not None:
        resolved_tmp_dir = root_path(tmp_dir)
        resolved_tmp_dir.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(resolved_tmp_dir)
    if env_defaults:
        for key, value in env_defaults.items():
            env.setdefault(key, str(value))
    if env_overrides:
        env.update({key: str(value) for key, value in env_overrides.items()})

    command_tuple = tuple(str(part) for part in command)
    status = "ok"
    error = ""
    started = time.perf_counter()
    with resolved_log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command_tuple) + "\n\n")
        completed = None
        try:
            completed = subprocess.run(
                command_tuple,
                cwd=root_path(cwd),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
            log.write(completed.stdout or "")
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            error = f"timeout_sec={timeout_sec}"
            if exc.stdout:
                log.write(str(exc.stdout))
            if exc.stderr:
                log.write(str(exc.stderr))
        if completed is not None and completed.returncode != 0:
            status = "failed"
            error = f"returncode={completed.returncode}"
    return LoggedSubprocessResult(status, error, time.perf_counter() - started, command_tuple)


def run_star_uvt_feature_trainer_subprocess(
    *,
    config_path: str | Path,
    log_path: str | Path,
    python: str | Path,
    timeout_sec: int | None,
    tmp_dir: str | Path | None = None,
    env_defaults: Mapping[str, str | Path | int | float] | None = None,
    env_overrides: Mapping[str, str | Path | int | float] | None = None,
) -> LoggedSubprocessResult:
    return run_logged_subprocess(
        (python, TRAIN_DISPATCHER, root_path(config_path)),
        log_path=log_path,
        cwd=ROOT,
        timeout_sec=timeout_sec,
        pythonpath=(TRAIN_ROOT, STAR_UVT_ROOT),
        env_defaults=env_defaults,
        env_overrides=env_overrides,
        tmp_dir=tmp_dir,
    )


__all__ = [
    "LoggedSubprocessResult",
    "ROOT",
    "STAR_UVT_FEATURE_TRAINER",
    "STAR_UVT_ROOT",
    "TRAIN_DISPATCHER",
    "TRAIN_ROOT",
    "distribution_stats",
    "fmt_cell",
    "fmt_pair",
    "load_report_json",
    "load_optional_report_json_or_error",
    "load_optional_report_json",
    "mean_timing_without_first",
    "read_report_csv",
    "root_path",
    "run_logged_subprocess",
    "run_star_uvt_feature_trainer_subprocess",
    "split_csv_floats",
    "split_csv_ints",
    "split_csv_strings",
    "summary_stats",
    "write_report_csv",
    "write_report_json",
    "write_report_text",
]
