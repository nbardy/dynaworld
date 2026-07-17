from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from config_utils import load_config_file


@dataclass(frozen=True)
class TrainerEntry:
    module: str
    runner: str = "run_training"
    resolver: str = "resolve_config"
    trainer_class: str | None = None


@dataclass(frozen=True)
class ExternalTrainerEntry:
    launcher: str
    note: str


TRAINER_BY_ARCH = {
    "tokengs": TrainerEntry("token_gs_trainer"),
    "tokengs_video_implicit_camera": TrainerEntry("token_gs_trainer"),
    "tokengs_video_known_camera": TrainerEntry("token_gs_trainer"),
    "precomputed_feature_implicit_camera": TrainerEntry(
        "precomputed_feature_trainer",
        resolver="PrecomputedFeatureImplicitTrainer.resolve_config",
        trainer_class="PrecomputedFeatureImplicitTrainer",
    ),
    "ltx_feature_implicit_camera": TrainerEntry(
        "precomputed_feature_trainer",
        resolver="PrecomputedFeatureImplicitTrainer.resolve_config",
        trainer_class="PrecomputedFeatureImplicitTrainer",
    ),
    "wan_vace_feature_implicit_camera": TrainerEntry(
        "precomputed_feature_trainer",
        resolver="PrecomputedFeatureImplicitTrainer.resolve_config",
        trainer_class="PrecomputedFeatureImplicitTrainer",
    ),
    "powerfoam_direct": TrainerEntry("powerfoam_direct_trainer"),
    "powerfoam_metal": TrainerEntry("powerfoam_metal_trainer"),
    "dynamic_powerfoam_metal": TrainerEntry("dynamic_powerfoam_metal_trainer"),
    "dynamic_gauge_foam": TrainerEntry("dynamic_gauge_foam_trainer"),
    "star_uvt_video_overfit": TrainerEntry("star_uvt_video_trainer"),
    "star_uvt_feature_overfit": TrainerEntry("star_uvt_feature_overfit_trainer"),
    "star_uvt_feature_rgb_probe": TrainerEntry(
        "star_uvt_feature_rgb_probe_trainer",
        runner="run_probe",
    ),
    "star_uvt_rendered_feature_rgb_probe": TrainerEntry(
        "star_uvt_rendered_feature_rgb_probe_trainer",
        runner="run_probe",
    ),
    "multicam_precomputed_feature_implicit_camera": TrainerEntry(
        "multicam_precomputed_trainer",
        resolver="MulticamPrecomputedFeatureImplicitTrainer.resolve_config",
        trainer_class="MulticamPrecomputedFeatureImplicitTrainer",
    ),
    "mixed_same_heldout_precomputed_feature_implicit_camera": TrainerEntry(
        "mixed_same_heldout_trainer",
        resolver="MixedSameHeldoutPrecomputedFeatureTrainer.resolve_config",
        trainer_class="MixedSameHeldoutPrecomputedFeatureTrainer",
    ),
    "multicam_relative_pose_implicit_camera": TrainerEntry(
        "multicam_relative_pose_trainer",
        resolver="MulticamRelativePoseImplicitTrainer.resolve_config",
        trainer_class="MulticamRelativePoseImplicitTrainer",
    ),
}


EXTERNAL_TRAINER_BY_ARCH = {
    "gauge_fields_material_surfel": ExternalTrainerEntry(
        launcher="research_experiments/gauge_fields/train.py",
        note="Gauge-field research CLI; keep outside src/train/train.py until its argparse-only runner is refactored.",
    ),
    "splat_baseline_free_dynamic_3dgs": ExternalTrainerEntry(
        launcher="research_experiments/gauge_fields/train_splat_baseline.py",
        note="Gauge-field 3DGS baseline CLI; uses the gauge experiment data bundle and local artifact layout.",
    ),
    "splat_baseline_static_3dgs": ExternalTrainerEntry(
        launcher="research_experiments/gauge_fields/train_splat_baseline.py",
        note="Gauge-field 3DGS baseline CLI; static/dynamic behavior is selected by config.",
    ),
}


def _expected_arches() -> str:
    return ", ".join(sorted(TRAINER_BY_ARCH))


def _external_arch_message(arch: str) -> str:
    entry = EXTERNAL_TRAINER_BY_ARCH[arch]
    return (
        f"arch={arch!r} is a checked-in external research trainer config, "
        f"not a src/train/train.py route. Launch it with {entry.launcher}. {entry.note}"
    )


def _dotted_attr(root: Any, name: str) -> Any:
    value = root
    for part in name.split("."):
        value = getattr(value, part)
    return value


def _entry_module(entry: TrainerEntry) -> Any:
    return import_module(entry.module)


def _entry_resolver(entry: TrainerEntry) -> Callable[[dict[str, Any]], dict[str, Any]]:
    resolver = _dotted_attr(_entry_module(entry), entry.resolver)
    if not callable(resolver):
        raise TypeError(f"Resolver {entry.module}.{entry.resolver} is not callable.")
    return resolver


def _entry_runner(entry: TrainerEntry) -> Callable[[dict[str, Any]], Any]:
    runner = _dotted_attr(_entry_module(entry), entry.runner)
    if not callable(runner):
        raise TypeError(f"Runner {entry.module}.{entry.runner} is not callable.")
    return runner


def config_arch(config: dict[str, Any], config_path: str | Path = "<config>") -> str:
    arch = config.get("arch")
    if arch is None:
        raise ValueError(f"Missing top-level 'arch' in {config_path}.")
    return str(arch).lower()


def trainer_entry_for_arch(arch: str, config_path: str | Path = "<config>") -> TrainerEntry:
    normalized = str(arch).lower()
    if normalized in EXTERNAL_TRAINER_BY_ARCH:
        raise ValueError(
            f"{_external_arch_message(normalized)} Supported train.py arches: {_expected_arches()}."
        )
    if normalized not in TRAINER_BY_ARCH:
        raise ValueError(
            f"Unsupported arch={normalized!r} in {config_path}. Expected one of: {_expected_arches()}. "
            f"External research arches: {', '.join(sorted(EXTERNAL_TRAINER_BY_ARCH))}."
        )
    return TRAINER_BY_ARCH[normalized]


def load_config_and_entry(config_path: str | Path) -> tuple[dict[str, Any], TrainerEntry]:
    path = Path(config_path)
    config = load_config_file(path)
    return config, trainer_entry_for_arch(config_arch(config, path), path)


def trainer_entry_for_config(config_path: str | Path) -> TrainerEntry:
    _config, entry = load_config_and_entry(config_path)
    return entry


def resolve_config_for_arch(config: dict[str, Any], config_path: str | Path = "<config>") -> dict[str, Any]:
    entry = trainer_entry_for_arch(config_arch(config, config_path), config_path)
    return _entry_resolver(entry)(config)


def trainer_class_for_config(config: dict[str, Any], config_path: str | Path = "<config>") -> type[Any]:
    entry = trainer_entry_for_arch(config_arch(config, config_path), config_path)
    module = _entry_module(entry)
    factory = getattr(module, "trainer_class_for_config", None)
    if callable(factory):
        return factory(config)
    if entry.trainer_class is not None:
        trainer_cls = _dotted_attr(module, entry.trainer_class)
        if not isinstance(trainer_cls, type):
            raise TypeError(f"Trainer class {entry.module}.{entry.trainer_class} is not a class.")
        return trainer_cls
    raise ValueError(
        f"arch={config_arch(config, config_path)!r} in {config_path} does not expose a trainer class factory. "
        f"Use trainer_registry.run_config(...) for {entry.module}.{entry.runner} routes."
    )


def instantiate_trainer_for_config(config: dict[str, Any], config_path: str | Path = "<config>") -> Any:
    return trainer_class_for_config(config, config_path)(config)


def run_config_dict(config: dict[str, Any], config_path: str | Path = "<config>") -> Any:
    entry = trainer_entry_for_arch(config_arch(config, config_path), config_path)
    return _entry_runner(entry)(config)


def run_config(config_path: str | Path) -> Any:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    config = load_config_file(path)
    return run_config_dict(config, path)
