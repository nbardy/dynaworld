from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import star_uvt_runtime


def test_ensure_star_uvt_on_path_can_keep_rgb_star_path_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "dynaworld"
    star_root = root / "third_party" / "star_uvt_v0"
    star_root.mkdir(parents=True)
    monkeypatch.setattr(star_uvt_runtime, "DYNAWORLD_ROOT", root)
    monkeypatch.setattr(star_uvt_runtime, "STAR_UVT_ROOT", star_root)
    monkeypatch.setattr(sys, "path", [])

    star_uvt_runtime.ensure_star_uvt_on_path(include_dynaworld_root=False)

    assert sys.path == [str(star_root)]


def test_ensure_star_uvt_on_path_adds_dynaworld_before_star_for_feature_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "dynaworld"
    star_root = root / "third_party" / "star_uvt_v0"
    star_root.mkdir(parents=True)
    monkeypatch.setattr(star_uvt_runtime, "DYNAWORLD_ROOT", root)
    monkeypatch.setattr(star_uvt_runtime, "STAR_UVT_ROOT", star_root)
    monkeypatch.setattr(sys, "path", [])

    star_uvt_runtime.ensure_star_uvt_on_path()

    assert sys.path == [str(star_root), str(root)]


def test_runtime_device_and_psnr_helpers_match_trainer_contract() -> None:
    assert star_uvt_runtime.resolve_device("cpu") == torch.device("cpu")
    assert star_uvt_runtime.psnr_from_loss(0.01) == pytest.approx(20.0)
