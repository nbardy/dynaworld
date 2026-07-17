from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from config_utils import load_config_file
from trainer_registry import (
    EXTERNAL_TRAINER_BY_ARCH,
    TRAINER_BY_ARCH,
    TrainerEntry,
    instantiate_trainer_for_config,
    resolve_config_for_arch,
    run_config_dict,
    trainer_class_for_config,
    trainer_entry_for_arch,
)


F32_SINGLE_CAM_CONFIG = "src/train_configs/local_mac_unconditioned_tokens_features_F32_alpha_400step.jsonc"
F32_MULTICAM_CONFIG = "src/train_configs/local_mac_ultimate_features_F32_vjepa_multicam_256px_8192splats_alpha.jsonc"


def _config_arches() -> dict[str, list[Path]]:
    arches: dict[str, list[Path]] = {}
    pattern = re.compile(r'"arch"\s*:\s*"([^"]+)"')
    for path in sorted(Path("src/train_configs").glob("*.json*")):
        match = pattern.search(path.read_text())
        if match is None:
            continue
        arches.setdefault(match.group(1).lower(), []).append(path)
    return arches


def test_all_checked_in_train_config_arches_are_registered_or_explicitly_external() -> None:
    arches = _config_arches()
    accounted = set(TRAINER_BY_ARCH) | set(EXTERNAL_TRAINER_BY_ARCH)

    missing = sorted(set(arches) - accounted)

    assert missing == []


def test_star_uvt_feature_rgb_probe_arch_routes_to_probe_runner() -> None:
    entry = trainer_entry_for_arch("star_uvt_feature_rgb_probe")

    assert entry.module == "star_uvt_feature_rgb_probe_trainer"
    assert entry.runner == "run_probe"


def test_star_uvt_feature_overfit_arch_routes_to_owner_module() -> None:
    entry = trainer_entry_for_arch("star_uvt_feature_overfit")

    assert entry.module == "star_uvt_feature_overfit_trainer"
    assert entry.runner == "run_training"


def test_powerfoam_metal_arch_routes_to_owner_module() -> None:
    entry = trainer_entry_for_arch("powerfoam_metal")

    assert entry.module == "powerfoam_metal_trainer"
    assert entry.runner == "run_training"


def test_powerfoam_direct_arch_routes_to_owner_module() -> None:
    entry = trainer_entry_for_arch("powerfoam_direct")

    assert entry.module == "powerfoam_direct_trainer"
    assert entry.runner == "run_training"


def test_dynamic_powerfoam_metal_arch_routes_to_owner_module() -> None:
    entry = trainer_entry_for_arch("dynamic_powerfoam_metal")

    assert entry.module == "dynamic_powerfoam_metal_trainer"
    assert entry.runner == "run_training"


def test_dynamic_gauge_foam_arch_routes_to_owner_module() -> None:
    entry = trainer_entry_for_arch("dynamic_gauge_foam")

    assert entry.module == "dynamic_gauge_foam_trainer"
    assert entry.runner == "run_training"


def test_star_uvt_rendered_feature_rgb_probe_arch_routes_to_probe_runner() -> None:
    entry = trainer_entry_for_arch("star_uvt_rendered_feature_rgb_probe")

    assert entry.module == "star_uvt_rendered_feature_rgb_probe_trainer"
    assert entry.runner == "run_probe"


def test_external_research_arch_reports_launcher_instead_of_generic_unknown() -> None:
    with pytest.raises(ValueError, match="research_experiments/gauge_fields/train.py"):
        trainer_entry_for_arch("gauge_fields_material_surfel")


def test_resolve_config_for_arch_uses_registered_module_resolver() -> None:
    cfg = load_config_file(F32_SINGLE_CAM_CONFIG)

    resolved = resolve_config_for_arch(cfg, F32_SINGLE_CAM_CONFIG)

    assert resolved["model"]["feature_dim"] == 32
    assert resolved["logging"]["wandb_enabled"] is True


def test_resolve_config_for_arch_uses_registered_classmethod_resolver() -> None:
    cfg = load_config_file(F32_MULTICAM_CONFIG)

    resolved = resolve_config_for_arch(cfg, F32_MULTICAM_CONFIG)

    assert resolved["model"]["video_encoder_backend"] == "precomputed"
    assert resolved["features"]["extractor"] == "vjepa_torchhub"


def test_trainer_class_for_config_routes_legacy_class_factory() -> None:
    cfg = load_config_file(F32_SINGLE_CAM_CONFIG)

    trainer_cls = trainer_class_for_config(cfg, F32_SINGLE_CAM_CONFIG)

    assert trainer_cls.__name__ == "Trainer"


def test_trainer_class_for_config_routes_class_based_precomputed_trainers() -> None:
    single_cfg = {
        "arch": "precomputed_feature_implicit_camera",
    }
    multicam_cfg = {
        "arch": "multicam_precomputed_feature_implicit_camera",
    }

    assert trainer_class_for_config(single_cfg).__name__ == "PrecomputedFeatureImplicitTrainer"
    assert trainer_class_for_config(multicam_cfg).__name__ == "MulticamPrecomputedFeatureImplicitTrainer"


def test_instantiate_trainer_for_config_uses_registry_class(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeTrainer:
        def __init__(self, config: dict[str, object]) -> None:
            calls.append(config)

    monkeypatch.setitem(sys.modules, "_fake_trainer_registry_class", SimpleNamespace(FakeTrainer=FakeTrainer))
    monkeypatch.setitem(
        TRAINER_BY_ARCH,
        "_fake_class_arch",
        TrainerEntry("_fake_trainer_registry_class", trainer_class="FakeTrainer"),
    )

    instance = instantiate_trainer_for_config({"arch": "_fake_class_arch", "value": 11}, "fake.jsonc")

    assert isinstance(instance, FakeTrainer)
    assert calls == [{"arch": "_fake_class_arch", "value": 11}]


def test_run_config_dict_dispatches_registered_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_runner(config: dict[str, object]) -> dict[str, object]:
        calls.append(config)
        return {"ok": True, "arch": config["arch"]}

    monkeypatch.setitem(sys.modules, "_fake_trainer_registry_runner", SimpleNamespace(fake_runner=fake_runner))
    monkeypatch.setitem(
        TRAINER_BY_ARCH,
        "_fake_arch",
        TrainerEntry("_fake_trainer_registry_runner", runner="fake_runner"),
    )

    result = run_config_dict({"arch": "_fake_arch", "value": 7}, "fake.jsonc")

    assert result == {"ok": True, "arch": "_fake_arch"}
    assert calls == [{"arch": "_fake_arch", "value": 7}]
