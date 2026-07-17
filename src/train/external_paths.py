from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"


def ensure_sys_path(*paths: str | Path, require_exists: bool = False) -> list[Path]:
    resolved_paths: list[Path] = []
    for path in paths:
        resolved = Path(path)
        if require_exists and not resolved.exists():
            continue
        resolved_paths.append(resolved)
        path_str = str(resolved)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return resolved_paths


def third_party_path(name: str) -> Path:
    return THIRD_PARTY_ROOT / name


def ensure_third_party_path(name: str, *, require_exists: bool = False) -> Path:
    path = third_party_path(name)
    ensure_sys_path(path, require_exists=require_exists)
    return path


def ensure_module_path(package_name: str, path: str | Path, *, missing_message: str | None = None) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise RuntimeError(missing_message or f"module path not found: {resolved}")
    existing_module = sys.modules.get(package_name)
    if existing_module is not None:
        origin_raw = getattr(existing_module, "__file__", None)
        if origin_raw is not None:
            origin = Path(origin_raw).resolve()
            if resolved.resolve() not in origin.parents:
                raise RuntimeError(f"{package_name!r} is already imported from {origin}, not {resolved}.")
    ensure_sys_path(resolved)
    return resolved


__all__ = [
    "PROJECT_ROOT",
    "THIRD_PARTY_ROOT",
    "ensure_module_path",
    "ensure_sys_path",
    "ensure_third_party_path",
    "third_party_path",
]
