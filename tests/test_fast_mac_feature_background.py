import pytest

from renderers.fast_mac import FastMacRendererConfig, _make_feature_config


def _fast_mac_config(values):
    return FastMacRendererConfig.from_mapping(
        values,
        fallback_tile_size=16,
        fallback_alpha_threshold=1.0 / 255.0,
    )


def test_feature_background_defaults_to_zero_without_reusing_rgb_background():
    config = _fast_mac_config({"background": [1.0, 1.0, 1.0]})

    assert config.background == (1.0, 1.0, 1.0)
    raster_config = _make_feature_config(config, height=8, width=8, feature_dim=32)
    assert raster_config.background == (0.0,)


def test_feature_background_accepts_scalar_or_explicit_feature_vector():
    scalar_config = _fast_mac_config({"feature_background": 0.25})
    scalar_raster_config = _make_feature_config(scalar_config, height=8, width=8, feature_dim=32)
    assert scalar_raster_config.background == (0.25,)

    feature_background = [0.0, 0.1, 0.2, 0.3]
    vector_config = _fast_mac_config({"feature_background": feature_background})
    vector_raster_config = _make_feature_config(vector_config, height=8, width=8, feature_dim=4)
    assert vector_raster_config.background == tuple(feature_background)


def test_feature_background_rejects_wrong_vector_length_for_feature_dim():
    config = _fast_mac_config({"feature_background": [0.0, 0.1, 0.2]})

    with pytest.raises(ValueError, match="feature_dim=32"):
        _make_feature_config(config, height=8, width=8, feature_dim=32)
