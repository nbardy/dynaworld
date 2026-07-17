from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from config_utils import load_config_file


Config = dict[str, Any]
ConfigInput = Config | str | Path | None
ResultT = TypeVar("ResultT")


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in parse_csv_strings(value)]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def run_config_or_path(config: Config | str | Path, runner: Callable[[Config], ResultT]) -> ResultT:
    if isinstance(config, (str, Path)):
        return runner(load_config_file(config))
    return runner(config)


def run_config_arg(
    runner: Callable[[Config], ResultT],
    *,
    usage: str,
    argv: Sequence[str] | None = None,
) -> ResultT:
    args = sys.argv if argv is None else argv
    if len(args) != 2:
        raise SystemExit(usage)
    return runner(load_config_file(args[1]))


def run_config_main(
    config: ConfigInput,
    runner: Callable[[Config], ResultT],
    *,
    usage: str,
    argv: Sequence[str] | None = None,
) -> ResultT:
    if config is None:
        return run_config_arg(runner, usage=usage, argv=argv)
    return run_config_or_path(config, runner)


def run_path_arg(
    runner: Callable[[str], ResultT],
    *,
    usage: str,
    argv: Sequence[str] | None = None,
) -> ResultT:
    args = sys.argv if argv is None else argv
    if len(args) != 2:
        raise SystemExit(usage)
    return runner(str(args[1]))


__all__ = [
    "ConfigInput",
    "parse_csv_ints",
    "parse_csv_strings",
    "run_config_arg",
    "run_config_main",
    "run_config_or_path",
    "run_path_arg",
]
