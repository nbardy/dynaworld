from __future__ import annotations

import torch

from star_uvt_colorizers import build_default_feature_colorizer, build_feature_colorizer, set_module_trainable


def _colorize_cfg(hidden_dim: int | None) -> dict[str, object]:
    return {
        "hidden_dim": hidden_dim,
        "activation": "sigmoid",
        "pre_norm": False,
        "weight_init": "kaiming",
        "weight_init_gain": 1.0,
    }


def test_build_feature_colorizer_preserves_single_layer_none_hidden_dim() -> None:
    colorizer = build_feature_colorizer(_colorize_cfg(None), feature_dim=5, device=torch.device("cpu"))
    features = torch.randn(2, 5, 3, 4)

    out = colorizer(features)

    assert colorizer.hidden_dim is None
    assert out.shape == (2, 3, 3, 4)
    assert out.device.type == "cpu"


def test_build_default_feature_colorizer_matches_feature_tube_default() -> None:
    colorizer = build_default_feature_colorizer(feature_dim=7, device=torch.device("cpu"))
    features = torch.randn(2, 7, 3, 4)

    out = colorizer(features)

    assert colorizer.hidden_dim is None
    assert colorizer.activation == "sigmoid"
    assert colorizer.weight_init == "kaiming"
    assert colorizer.weight_init_gain == 4.0
    assert colorizer.pre_norm is not None
    assert out.shape == (2, 3, 3, 4)
    assert out.device.type == "cpu"


def test_set_module_trainable_toggles_requires_grad_and_mode() -> None:
    colorizer = build_feature_colorizer(_colorize_cfg(8), feature_dim=5, device=torch.device("cpu"))

    set_module_trainable(colorizer, False)

    assert not colorizer.training
    assert all(not param.requires_grad for param in colorizer.parameters())

    set_module_trainable(colorizer, True)

    assert colorizer.training
    assert all(param.requires_grad for param in colorizer.parameters())
