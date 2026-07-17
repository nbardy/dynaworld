from __future__ import annotations

from pathlib import Path

import torch

from powerfoam_checkpoints import save_powerfoam_checkpoint


def test_save_powerfoam_checkpoint_preserves_direct_minimal_payload(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 3)
    path = tmp_path / "checkpoint_final.pt"

    save_powerfoam_checkpoint(path, model, {"logging": {"output_dir": tmp_path}})

    payload = torch.load(path, map_location="cpu")

    assert set(payload) == {"model", "config"}
    assert isinstance(payload["model"], dict)
    assert payload["config"] == {"logging": {"output_dir": str(tmp_path)}}


def test_save_powerfoam_checkpoint_keeps_metric_payload_when_step_is_given(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 3)
    path = tmp_path / "checkpoint_best.pt"

    save_powerfoam_checkpoint(
        path,
        model,
        {"train": {"steps": 4}},
        step=3,
        metrics={"eval_psnr": 12.5},
        best_metric_name="eval_psnr",
        best_metric_value=12.5,
    )

    payload = torch.load(path, map_location="cpu")

    assert isinstance(payload["model"], dict)
    assert payload["config"] == {"train": {"steps": 4}}
    assert payload["step"] == 3
    assert payload["metrics"] == {"eval_psnr": 12.5}
    assert payload["best_metric_name"] == "eval_psnr"
    assert payload["best_metric_value"] == 12.5
