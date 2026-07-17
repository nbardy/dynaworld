from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

import common
from common import parse_csv_strings, run_gauge_matrix
from common import write_checkpoint


def test_write_checkpoint_resolves_path_and_uses_atomic_save(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[dict[str, object], Path]] = []

    def fake_atomic_torch_save(payload: dict[str, object], path: Path) -> None:
        calls.append((payload, path))

    monkeypatch.setattr(common, "atomic_torch_save", fake_atomic_torch_save)
    payload = {"model": {"weight": torch.tensor([1.0])}}

    saved_path = write_checkpoint(tmp_path / "checkpoint.pt", payload)

    assert saved_path == tmp_path / "checkpoint.pt"
    assert calls == [(payload, tmp_path / "checkpoint.pt")]


def test_run_gauge_matrix_uses_shared_csv_only_filter(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, *, cwd, check):
        calls.append(list(cmd))

    monkeypatch.setattr(common.subprocess, "run", fake_run)

    args = SimpleNamespace(
        output_root=str(tmp_path),
        steps=3,
        device="cpu",
        no_wandb=True,
        only=" keep , missing ",
    )
    runs = [
        {"name": "skip", "script": Path("skip.py"), "config": Path("skip.jsonc")},
        {"name": "keep", "script": Path("keep.py"), "config": Path("keep.jsonc")},
    ]

    run_gauge_matrix(runs, args)

    assert len(calls) == 1
    assert "keep.py" in calls[0]
    assert "skip.py" not in calls[0]
    assert (tmp_path / "keep" / "wall_clock.json").exists()


def test_parse_csv_strings_trims_empty_values() -> None:
    assert parse_csv_strings(" a, ,b , c ") == ["a", "b", "c"]
