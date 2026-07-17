from __future__ import annotations

from pathlib import Path

import torch

from star_uvt_checkpoints import (
    load_feature_rgb_probe_checkpoint,
    load_star_model_from_training_checkpoint,
    save_feature_rgb_probe_checkpoint,
    save_rendered_feature_rgb_probe_checkpoint,
)
from star_uvt_colorizers import build_feature_colorizer


def test_load_star_model_from_training_checkpoint_freezes_model_and_returns_row_metadata(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 3)
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.bias.fill_(-0.5)
    path = tmp_path / "star_feature_overfit.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "steps": 17,
            "row": {
                "pass": True,
                "end_feature_target_loss": 0.125,
                "end_rgb_probe_psnr": 31.5,
            },
        },
        path,
    )

    target = torch.nn.Linear(2, 3)
    state = load_star_model_from_training_checkpoint(
        path,
        model=target,
        device=torch.device("cpu"),
        freeze_model=True,
    )

    assert state == {
        "path": str(path),
        "steps": 17,
        "row_pass": True,
        "row_end_feature_target_loss": 0.125,
        "row_end_rgb_probe_psnr": 31.5,
    }
    torch.testing.assert_close(target.weight, source.weight)
    torch.testing.assert_close(target.bias, source.bias)
    assert not target.training
    assert all(not param.requires_grad for param in target.parameters())


def test_feature_rgb_probe_checkpoint_roundtrips_colorizer_and_metadata(tmp_path: Path) -> None:
    colorize_cfg = {
        "hidden_dim": None,
        "activation": "sigmoid",
        "pre_norm": False,
        "weight_init": "kaiming",
        "weight_init_gain": 1.0,
    }
    cfg = {
        "feature_uvt": {"feature_dim": 3},
        "colorize": colorize_cfg,
    }
    colorizer = build_feature_colorizer(colorize_cfg, feature_dim=3, device=torch.device("cpu"))
    features = torch.randn(2, 3, 4, 5)
    expected = colorizer(features)
    path = tmp_path / "rgb_probe.pt"

    save_feature_rgb_probe_checkpoint(
        path,
        colorizer=colorizer,
        cfg=cfg,
        feature_target_meta={"source": "fixture"},
        target_grid_shape=(2, 3, 4, 5),
        target_rgb_shape=(2, 3, 4, 5),
        grid_loss=0.25,
        full_loss=0.5,
    )

    loaded, meta = load_feature_rgb_probe_checkpoint(path, device=torch.device("cpu"), feature_dim=3)

    torch.testing.assert_close(loaded(features), expected)
    assert meta["checkpoint"] == str(path)
    assert meta["feature_dim"] == 3
    assert meta["hidden_dim"] is None
    assert meta["grid_loss"] == 0.25
    assert meta["full_loss"] == 0.5
    assert meta["target_grid_shape"] == [2, 3, 4, 5]
    assert meta["target_rgb_shape"] == [2, 3, 4, 5]
    assert not loaded.training
    assert all(not param.requires_grad for param in loaded.parameters())


def test_rendered_feature_rgb_probe_checkpoint_serializes_probe_payload(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 3)
    colorizer = torch.nn.Conv2d(3, 3, kernel_size=1)
    optimizer = torch.optim.Adam(colorizer.parameters(), lr=0.125)
    path = tmp_path / "rendered_probe.pt"

    save_rendered_feature_rgb_probe_checkpoint(
        path,
        model=model,
        colorizer=colorizer,
        optimizer=optimizer,
        cfg={"probe": {"steps": 2}, "path": tmp_path / "video.mp4"},
        resume_state={"path": tmp_path / "source.pt", "loaded": True},
        colorizer_init_state={"path": None, "loaded": False},
        train_star_model=False,
        sparse_sample_loss=0.125,
        full_loss=0.25,
    )

    payload = torch.load(path, map_location="cpu")

    assert payload["model"] is None
    assert isinstance(payload["colorizer"], dict)
    assert isinstance(payload["optimizer"], dict)
    assert payload["config"] == {"probe": {"steps": 2}, "path": str(tmp_path / "video.mp4")}
    assert payload["resume_state"] == {"path": str(tmp_path / "source.pt"), "loaded": True}
    assert payload["colorizer_init_state"] == {"path": None, "loaded": False}
    assert payload["sparse_sample_loss"] == 0.125
    assert payload["full_loss"] == 0.25
