from __future__ import annotations

import torch

from star_uvt_models import build_feature_tube_model
from star_uvt_render_configs import feature_tube_render_config_from_cfg


def _cfg(seed: int = 11) -> dict[str, object]:
    return {
        "data": {
            "max_frames": 3,
            "target_size": 16,
        },
        "feature_uvt": {
            "tube_count": 5,
            "feature_dim": 4,
            "alpha_threshold": 1.0 / 255.0,
            "max_alpha": 0.99,
        },
        "train": {"seed": seed},
        "probe": {"seed": seed + 1},
    }


def test_build_feature_tube_model_uses_configured_tube_count_and_device() -> None:
    cfg = _cfg()
    feature_config = feature_tube_render_config_from_cfg(cfg)

    model = build_feature_tube_model(cfg, feature_config, device=torch.device("cpu"))

    assert model.tube_count == 5
    assert model.config is feature_config
    assert model.raw_feature.shape == (5, 4)
    assert model.raw_feature.device.type == "cpu"


def test_build_feature_tube_model_supports_probe_seed_section() -> None:
    cfg = _cfg()
    feature_config = feature_tube_render_config_from_cfg(cfg)

    train_seed_model = build_feature_tube_model(cfg, feature_config, device=torch.device("cpu"))
    probe_seed_model = build_feature_tube_model(
        cfg,
        feature_config,
        device=torch.device("cpu"),
        seed_section="probe",
    )

    assert not torch.equal(train_seed_model.center_uv, probe_seed_model.center_uv)
