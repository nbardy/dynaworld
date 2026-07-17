from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from train_cli import (
    parse_csv_ints,
    parse_csv_strings,
    run_config_arg,
    run_config_main,
    run_config_or_path,
    run_path_arg,
)


def _write_config(path: Path) -> None:
    path.write_text('{"arch": "tokengs", "value": 7}\n')


def test_run_config_or_path_passes_config_dict() -> None:
    config = {"arch": "tokengs", "value": 3}
    seen: list[dict[str, Any]] = []

    result = run_config_or_path(config, lambda cfg: seen.append(cfg) or "ok")

    assert result == "ok"
    assert seen == [config]


def test_parse_csv_ints_strips_and_ignores_empty_items() -> None:
    assert parse_csv_ints(" 0, 2,,5 ") == [0, 2, 5]


def test_parse_csv_strings_strips_and_ignores_empty_items() -> None:
    assert parse_csv_strings(" alpha, beta,,gamma ") == ["alpha", "beta", "gamma"]


def test_run_config_or_path_loads_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.jsonc"
    _write_config(config_path)
    seen: list[dict[str, Any]] = []

    run_config_or_path(config_path, seen.append)

    assert seen == [{"arch": "tokengs", "value": 7}]


def test_run_config_arg_loads_argv_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.jsonc"
    _write_config(config_path)
    seen: list[dict[str, Any]] = []

    run_config_arg(seen.append, usage="Usage: train <config>", argv=["train.py", str(config_path)])

    assert seen == [{"arch": "tokengs", "value": 7}]


def test_run_config_arg_reports_usage_for_wrong_arity() -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_config_arg(lambda _cfg: None, usage="Usage: train <config>", argv=["train.py"])

    assert str(exc_info.value) == "Usage: train <config>"


def test_run_config_main_passes_config_dict() -> None:
    config = {"arch": "tokengs", "value": 3}
    seen: list[dict[str, Any]] = []

    run_config_main(config, seen.append, usage="Usage: train <config>")

    assert seen == [config]


def test_run_config_main_loads_path_or_argv(tmp_path: Path) -> None:
    config_path = tmp_path / "config.jsonc"
    _write_config(config_path)
    seen: list[dict[str, Any]] = []

    run_config_main(config_path, seen.append, usage="Usage: train <config>")
    run_config_main(
        None,
        seen.append,
        usage="Usage: train <config>",
        argv=["train.py", str(config_path)],
    )

    assert seen == [{"arch": "tokengs", "value": 7}] * 2


def test_run_path_arg_passes_raw_config_path() -> None:
    seen: list[str] = []

    run_path_arg(seen.append, usage="Usage: train <config>", argv=["train.py", "config.jsonc"])

    assert seen == ["config.jsonc"]


def test_run_path_arg_reports_usage_for_wrong_arity() -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_path_arg(lambda _path: None, usage="Usage: train <config>", argv=["train.py"])

    assert str(exc_info.value) == "Usage: train <config>"
