from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GAUGE_FIELDS = ROOT / "research_experiments" / "gauge_fields"


def _prioritize_path(path: Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def prefer_gauge_field_imports() -> None:
    _prioritize_path(GAUGE_FIELDS)
    if (train_module := sys.modules.get("train")) is not None:
        module_file = getattr(train_module, "__file__", "")
        if module_file and module_file.endswith("/src/train/train.py"):
            del sys.modules["train"]


for path in (
    ROOT,
    ROOT / "src",
    ROOT / "src" / "dataset_pipeline",
    ROOT / "src" / "train",
    GAUGE_FIELDS,
):
    _prioritize_path(path)
prefer_gauge_field_imports()
