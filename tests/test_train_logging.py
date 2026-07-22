from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from train_logging import (
    finish_wandb_run,
    flattened_scalar_payload,
    init_wandb_run,
    log_wandb_payload,
    log_wandb_run_payload,
    log_wandb_run_payload_lazy,
    mapped_metric_payload,
    scalar_payload,
    set_default_wandb_mode,
    should_log_from_config,
    should_log_image,
    should_log_scalar,
    should_log_step,
    should_log_video,
    wandb_run_lifecycle,
)


def _cfg() -> dict:
    return {
        "train": {"steps": 10},
        "logging": {
            "log_every": 3,
            "image_log_every": 4,
            "video_log_every": 5,
            "always_log_last_step": True,
        },
    }


def test_should_log_step_uses_interval_and_last_step() -> None:
    assert not should_log_step(2, 3, total_steps=10, always_log_last_step=True)
    assert should_log_step(3, 3, total_steps=10, always_log_last_step=True)
    assert should_log_step(10, 3, total_steps=10, always_log_last_step=True)


def test_should_log_step_can_skip_initial_media() -> None:
    assert not should_log_step(0, 1, total_steps=10, always_log_last_step=True, log_step_zero=False)
    assert should_log_step(0, 1, total_steps=10, always_log_last_step=True, log_step_zero=True)


def test_config_log_gates_use_named_logging_intervals() -> None:
    cfg = _cfg()

    assert should_log_scalar(cfg, 3)
    assert not should_log_image(cfg, 3)
    assert should_log_image(cfg, 4)
    assert should_log_video(cfg, 5)
    assert should_log_from_config(cfg, 10, "image_log_every")


def test_flattened_scalar_payload_recurses_and_skips_bool_flags() -> None:
    row = {
        "loss": 0.25,
        "passed": True,
        "nested": {
            "step": 4,
            "enabled": False,
            "deeper": {"psnr": 12.5},
        },
        "text": "ignored",
    }

    payload = flattened_scalar_payload("star_uvt", row)

    assert payload == {
        "star_uvt/loss": 0.25,
        "star_uvt/nested/step": 4,
        "star_uvt/nested/deeper/psnr": 12.5,
    }


def test_mapped_metric_payload_requires_or_skips_missing_metrics() -> None:
    metrics = {"eval_l1": 0.1, "eval_mse": 0.2}
    key_map = (
        ("eval_l1", "Eval/L1"),
        ("eval_mse", "Eval/MSE"),
        ("heldout_eval_l1", "Heldout/EvalL1"),
    )

    assert mapped_metric_payload(metrics, key_map, require=False) == {
        "Eval/L1": 0.1,
        "Eval/MSE": 0.2,
    }
    with pytest.raises(KeyError, match="heldout_eval_l1"):
        mapped_metric_payload(metrics, key_map)


def test_scalar_payload_builds_step_result_metrics() -> None:
    cfg = {
        "model": {"train_frame_count": 8, "size": 64},
        "render": {"render_size": 32},
    }
    camera_state = SimpleNamespace(
        fov_degrees=torch.tensor(55.0),
        radius=torch.tensor(2.5),
        rotation_delta=torch.zeros(2, 3),
        translation_delta=torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    result = SimpleNamespace(
        loss=torch.tensor(1.0),
        recon_loss=torch.tensor(0.7),
        camera_motion_loss=torch.tensor(0.01),
        camera_temporal_loss=torch.tensor(0.02),
        camera_global_loss=torch.tensor(0.03),
        bank_rate_loss=torch.tensor(0.04),
        sequence_frame_count=5,
        camera_state=camera_state,
        bank_rate_terms={"dynamic_alpha": torch.tensor(0.25)},
        aux_loss_terms={"same_view_recon": torch.tensor(0.6)},
    )

    payload = scalar_payload(cfg, result, train_sequence_count=2, eval_sequence_count=1)

    assert payload["Loss"] == pytest.approx(1.0)
    assert payload["Loss/Reconstruction"] == pytest.approx(0.7)
    assert payload["TrainFrameCount"] == 8
    assert payload["SequenceFrames"] == 5
    assert payload["TrainSequenceCount"] == 2
    assert payload["EvalSequenceCount"] == 1
    assert payload["InputSize"] == 64
    assert payload["RenderSize"] == 32
    assert payload["Camera/FOVDegrees"] == pytest.approx(55.0)
    assert payload["Camera/Radius"] == pytest.approx(2.5)
    assert payload["Camera/RotationDeltaMeanDegrees"] == pytest.approx(0.0)
    assert payload["Camera/TranslationDeltaMean"] == pytest.approx(2.5)
    assert payload["BankRate/dynamic_alpha"] == pytest.approx(0.25)
    assert payload["Loss/same_view_recon"] == pytest.approx(0.6)


def test_init_wandb_run_returns_none_when_disabled(monkeypatch: Any) -> None:
    def fail_init(**_: Any) -> None:
        raise AssertionError("wandb.init should not run when logging is disabled")

    monkeypatch.setattr("train_logging.wandb.init", fail_init)

    assert init_wandb_run({"logging": {"wandb_enabled": False}}) is None


def test_init_wandb_run_uses_shared_logging_config(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_init(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "wandb-run"

    monkeypatch.setattr("train_logging.wandb.init", fake_init)
    cfg = {
        "logging": {
            "wandb_enabled": True,
            "wandb_project": "dynaworld",
            "wandb_run_name": "shared-init-test",
            "wandb_tags": ["trainer"],
            "wandb_mode": "offline",
            "wandb_run_id": "stable123",
            "wandb_resume": "allow",
        },
        "output": {"checkpoint": Path("outputs/checkpoint.pt")},
    }

    assert init_wandb_run(cfg) == "wandb-run"
    assert calls == [
        {
            "project": "dynaworld",
            "name": "shared-init-test",
            "tags": ["trainer"],
            "mode": "offline",
            "id": "stable123",
            "resume": "allow",
            "config": {
                "logging": cfg["logging"],
                "output": {"checkpoint": "outputs/checkpoint.pt"},
            },
        }
    ]


def test_init_wandb_run_can_disable_dirty_diff_uploads(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("train_logging.wandb.init", lambda **kwargs: calls.append(kwargs) or "run")
    cfg = {
        "logging": {
            "wandb_enabled": True,
            "wandb_project": "dynaworld",
            "wandb_run_name": "paper",
            "wandb_tags": ["paper"],
            "wandb_mode": "offline",
            "wandb_disable_git": True,
            "wandb_disable_code": True,
        }
    }

    assert init_wandb_run(cfg) == "run"
    assert calls[0]["settings"].disable_git is True
    assert calls[0]["settings"].disable_code is True


def test_finish_wandb_run_skips_disabled_run(monkeypatch: Any) -> None:
    def fail_finish() -> None:
        raise AssertionError("wandb.finish should not run without an active run")

    monkeypatch.setattr("train_logging.wandb.finish", fail_finish)

    finish_wandb_run(None)


def test_finish_wandb_run_finishes_active_run(monkeypatch: Any) -> None:
    calls: list[str] = []

    monkeypatch.setattr("train_logging.wandb.finish", lambda: calls.append("global"))

    class Run:
        def finish(self) -> None:
            calls.append("run")

    finish_wandb_run(Run())

    assert calls == ["run"]


def test_finish_wandb_run_can_finish_global_active_run(monkeypatch: Any) -> None:
    calls: list[str] = []

    class Run:
        def finish(self) -> None:
            calls.append("run")

    monkeypatch.setattr("train_logging.wandb.run", Run(), raising=False)

    finish_wandb_run()

    assert calls == ["run"]


def test_finish_wandb_run_falls_back_to_global_finish(monkeypatch: Any) -> None:
    calls: list[str] = []

    monkeypatch.setattr("train_logging.wandb.finish", lambda: calls.append("global"))

    finish_wandb_run(object())

    assert calls == ["global"]


def test_wandb_run_lifecycle_finishes_normal_path(monkeypatch: Any) -> None:
    calls: list[tuple[str, Any]] = []

    monkeypatch.setattr("train_logging.init_wandb_run", lambda cfg: calls.append(("init", cfg)) or "run")
    monkeypatch.setattr("train_logging.finish_wandb_run", lambda run: calls.append(("finish", run)))

    cfg = {"logging": {"wandb_enabled": True}}
    with wandb_run_lifecycle(cfg) as run:
        calls.append(("body", run))

    assert calls == [
        ("init", cfg),
        ("body", "run"),
        ("finish", "run"),
    ]


def test_wandb_run_lifecycle_finishes_exception_path(monkeypatch: Any) -> None:
    calls: list[tuple[str, Any]] = []

    monkeypatch.setattr("train_logging.init_wandb_run", lambda cfg: calls.append(("init", cfg)) or "run")
    monkeypatch.setattr("train_logging.finish_wandb_run", lambda run: calls.append(("finish", run)))

    cfg = {"logging": {"wandb_enabled": True}}
    with pytest.raises(RuntimeError, match="boom"):
        with wandb_run_lifecycle(cfg) as run:
            calls.append(("body", run))
            raise RuntimeError("boom")

    assert calls == [
        ("init", cfg),
        ("body", "run"),
        ("finish", "run"),
    ]


def test_log_wandb_payload_forwards_payload_and_step(monkeypatch: Any) -> None:
    calls: list[tuple[dict[str, Any], int | None]] = []

    def fake_log(payload: dict[str, Any], step: int | None = None) -> None:
        calls.append((payload, step))

    monkeypatch.setattr("train_logging.wandb.log", fake_log)

    payload = {"Loss": 1.0}
    log_wandb_payload(payload, step=7)

    assert calls == [({"Loss": 1.0}, 7)]
    assert calls[0][0] is not payload


def test_log_wandb_run_payload_forwards_explicit_run_and_step() -> None:
    calls: list[tuple[dict[str, Any], int | None]] = []

    class Run:
        def log(self, payload: dict[str, Any], step: int | None = None) -> None:
            calls.append((payload, step))

    payload = {"Loss": 1.0}
    log_wandb_run_payload(Run(), payload, step=7)

    assert calls == [({"Loss": 1.0}, 7)]
    assert calls[0][0] is not payload


def test_log_wandb_run_payload_skips_disabled_run() -> None:
    log_wandb_run_payload(None, {"Loss": 1.0}, step=7)


def test_log_wandb_run_payload_lazy_skips_factory_for_disabled_run() -> None:
    def fail_factory() -> dict[str, float]:
        raise AssertionError("payload factory should not run without a wandb run")

    log_wandb_run_payload_lazy(None, fail_factory, step=7)


def test_log_wandb_run_payload_lazy_builds_payload_once_for_explicit_run() -> None:
    calls: list[tuple[dict[str, Any], int | None]] = []
    factory_calls: list[str] = []

    class Run:
        def log(self, payload: dict[str, Any], step: int | None = None) -> None:
            calls.append((payload, step))

    def payload_factory() -> dict[str, float]:
        factory_calls.append("factory")
        return {"Loss": 1.0}

    log_wandb_run_payload_lazy(Run(), payload_factory, step=7)

    assert factory_calls == ["factory"]
    assert calls == [({"Loss": 1.0}, 7)]


def test_set_default_wandb_mode_preserves_existing_environment(monkeypatch: Any) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_SILENT", raising=False)

    set_default_wandb_mode("disabled", silent=True)

    assert os.environ["WANDB_MODE"] == "online"
    assert os.environ["WANDB_SILENT"] == "true"


def test_set_default_wandb_mode_can_leave_silent_unset(monkeypatch: Any) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_SILENT", raising=False)

    set_default_wandb_mode("disabled", silent=None)

    assert os.environ["WANDB_MODE"] == "disabled"
    assert "WANDB_SILENT" not in os.environ
