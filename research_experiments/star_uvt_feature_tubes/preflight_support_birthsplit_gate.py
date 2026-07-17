from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT / "src" / "train"
for path in (ROOT, SCRIPT_DIR, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config_utils import load_config_file, serialize_config_value  # noqa: E402
from report_artifacts import write_report_json, write_report_text  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_50step_media.jsonc"
)
DEFAULT_OUT_DIR = ROOT / "outputs/benchmarks/2026-05-25_star_uvt_birthsplit_r64_o04_50step_preflight"

EXPECTED_CONFIG_VALUES: tuple[tuple[tuple[str, ...], Any], ...] = (
    (("arch",), "star_uvt_feature_overfit"),
    (("feature_target", "enabled"), True),
    (("feature_target", "materialization"), "target_grid"),
    (("feature_target", "image_vjp_mode"), "analytic_sparse_grid_forward_batched"),
    (("feature_uvt", "render_mode"), "feature_direct_gradcache_reduce_vec4"),
    (("feature_uvt", "tile_capacity"), 128),
    (("feature_uvt", "tube_count"), 8192),
    (("support_birth_split", "enabled"), True),
    (("support_birth_split", "center_strategy"), "farthest_xy"),
    (("support_birth_split", "center_count"), 8),
    (("support_birth_split", "reallocate_tubes"), 32),
    (("support_birth_split", "support_shape"), "isotropic"),
    (("support_birth_split", "support_radius_px"), 64.0),
    (("support_birth_split", "support_precision_radius_px"), 64.0),
    (("support_birth_split", "support_radius_across_px"), 64.0),
    (("support_birth_split", "support_radius_along_px"), 64.0),
    (("support_birth_split", "temporal_radius_frames"), 64.0),
    (("support_birth_split", "opacity"), 0.4),
    (("support_birth_split", "target_point_source"), "uncovered_brightness"),
    (("train", "global_step_offset"), 1500),
    (("train", "resume_colorizer"), False),
    (("train", "resume_optimizer"), False),
    (("train", "require_loss_decrease"), True),
    (("train", "require_no_tile_overflow"), True),
)

REQUIRED_ARTIFACT_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("train.resume_checkpoint", ("train", "resume_checkpoint")),
    ("colorize.init_checkpoint", ("colorize", "init_checkpoint")),
    ("feature_target.rgb_probe_checkpoint", ("feature_target", "rgb_probe_checkpoint")),
    ("data.video_path", ("data", "video_path")),
)


@dataclass(frozen=True)
class PreflightCheck:
    id: str
    status: str
    required: bool
    message: str
    path: str | None = None
    expected: Any | None = None
    actual: Any | None = None


def root_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def _path_key(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _config_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(_path_key(path))
        value = value[key]
    return value


def _matches_expected(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1.0e-6
        except (TypeError, ValueError):
            return False
    return actual == expected


def _check_config_value(config: dict[str, Any], path: tuple[str, ...], expected: Any) -> PreflightCheck:
    check_id = _path_key(path)
    try:
        actual = _config_value(config, path)
    except KeyError:
        return PreflightCheck(
            id=check_id,
            status="missing",
            required=True,
            expected=expected,
            message="Required config key is missing.",
        )
    if _matches_expected(actual, expected):
        return PreflightCheck(
            id=check_id,
            status="ok",
            required=True,
            expected=expected,
            actual=actual,
            message="Config value matches the selected support gate.",
        )
    return PreflightCheck(
        id=check_id,
        status="mismatch",
        required=True,
        expected=expected,
        actual=actual,
        message="Config value does not match the selected support gate.",
    )


def _check_path_field(config: dict[str, Any], check_id: str, path: tuple[str, ...]) -> PreflightCheck:
    try:
        value = _config_value(config, path)
    except KeyError:
        return PreflightCheck(
            id=check_id,
            status="missing",
            required=True,
            message="Required artifact path key is missing from config.",
        )
    resolved = root_path(value)
    if resolved.exists():
        return PreflightCheck(
            id=check_id,
            status="ok",
            required=True,
            path=str(resolved),
            message="Required input artifact exists.",
        )
    return PreflightCheck(
        id=check_id,
        status="missing",
        required=True,
        path=str(resolved),
        message="Required input artifact is missing.",
    )


def _check_expected_steps(config: dict[str, Any], expected_steps: int) -> PreflightCheck:
    return _check_config_value(config, ("train", "steps"), int(expected_steps))


def _check_feature_cache(config: dict[str, Any], *, require_feature_cache: bool) -> PreflightCheck:
    try:
        value = _config_value(config, ("features", "cache_dir"))
    except KeyError:
        return PreflightCheck(
            id="features.cache_dir",
            status="missing",
            required=require_feature_cache,
            message="Feature cache path is absent from the config.",
        )
    cache_dir = root_path(value)
    if not cache_dir.exists():
        return PreflightCheck(
            id="features.cache_dir",
            status="missing" if require_feature_cache else "warning",
            required=require_feature_cache,
            path=str(cache_dir),
            message="Feature cache directory is missing; training may need a slow rebake/download.",
        )
    cached_files = sorted(cache_dir.glob("*.pt"))
    if not cached_files:
        return PreflightCheck(
            id="features.cache_files",
            status="missing" if require_feature_cache else "warning",
            required=require_feature_cache,
            path=str(cache_dir),
            message="Feature cache directory exists but has no .pt cache files.",
        )
    return PreflightCheck(
        id="features.cache_files",
        status="ok",
        required=require_feature_cache,
        path=str(cache_dir),
        actual=len(cached_files),
        message="Feature cache directory has at least one .pt cache file.",
    )


def _run_command(config_path: Path) -> str:
    config_arg = config_path if config_path.is_absolute() else config_path.relative_to(ROOT)
    return f"PYTHONPATH=src/train .venv/bin/python src/train/train.py {config_arg}"


def _status_for_checks(checks: list[PreflightCheck]) -> str:
    return "blocked" if any(check.required and check.status != "ok" for check in checks) else "ready"


def evaluate_preflight(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    expected_steps: int = 50,
    require_feature_cache: bool = False,
) -> dict[str, Any]:
    resolved_config_path = root_path(config_path)
    checks: list[PreflightCheck] = []
    if not resolved_config_path.exists():
        checks.append(
            PreflightCheck(
                id="config",
                status="missing",
                required=True,
                path=str(resolved_config_path),
                message="Config file is missing.",
            )
        )
        return {
            "status": "blocked",
            "benchmark": "star_uvt_birthsplit_r64_o04_50step_preflight",
            "config_path": str(resolved_config_path),
            "expected_steps": int(expected_steps),
            "checks": [asdict(check) for check in checks],
            "blocking_check_ids": ["config"],
            "warning_check_ids": [],
            "run_command": _run_command(resolved_config_path),
        }

    config = load_config_file(resolved_config_path)
    checks.append(
        PreflightCheck(
            id="config",
            status="ok",
            required=True,
            path=str(resolved_config_path),
            message="Config file exists and parses as JSONC.",
        )
    )
    checks.extend(_check_config_value(config, path, expected) for path, expected in EXPECTED_CONFIG_VALUES)
    checks.append(_check_expected_steps(config, expected_steps))
    checks.extend(_check_path_field(config, check_id, path) for check_id, path in REQUIRED_ARTIFACT_FIELDS)
    checks.append(_check_feature_cache(config, require_feature_cache=require_feature_cache))

    blocking_ids = [check.id for check in checks if check.required and check.status != "ok"]
    warning_ids = [check.id for check in checks if check.status == "warning"]
    return {
        "status": _status_for_checks(checks),
        "benchmark": "star_uvt_birthsplit_r64_o04_50step_preflight",
        "config_path": str(resolved_config_path),
        "expected_steps": int(expected_steps),
        "require_feature_cache": bool(require_feature_cache),
        "checks": [serialize_config_value(asdict(check)) for check in checks],
        "blocking_check_ids": blocking_ids,
        "warning_check_ids": warning_ids,
        "run_command": _run_command(resolved_config_path),
    }


def _md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# STAR UVT Birth/Split Preflight",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"Config: `{payload['config_path']}`",
        f"Expected steps: `{payload['expected_steps']}`",
        "",
        "Run command:",
        "",
        "```bash",
        str(payload["run_command"]),
        "```",
        "",
    ]
    if payload["blocking_check_ids"]:
        lines.extend(
            [
                "Blocking checks:",
                "",
                ", ".join(f"`{check_id}`" for check_id in payload["blocking_check_ids"]),
                "",
            ]
        )
    if payload["warning_check_ids"]:
        lines.extend(
            [
                "Warnings:",
                "",
                ", ".join(f"`{check_id}`" for check_id in payload["warning_check_ids"]),
                "",
            ]
        )
    lines.extend(
        [
            "| Check | Status | Required | Path | Expected | Actual | Message |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for check in payload["checks"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    _md(check["id"]),
                    _md(check["status"]),
                    _md(check["required"]),
                    _md(check.get("path")),
                    _md(check.get("expected")),
                    _md(check.get("actual")),
                    _md(check["message"]),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def run_report(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUT_DIR,
    expected_steps: int = 50,
    require_feature_cache: bool = False,
) -> dict[str, Any]:
    payload = evaluate_preflight(
        config_path,
        expected_steps=expected_steps,
        require_feature_cache=require_feature_cache,
    )
    output_path = root_path(output_dir)
    payload["summary_json"] = str(write_report_json(output_path / "summary.json", payload))
    payload["summary_md"] = str(write_report_text(output_path / "summary.md", markdown_report(payload)))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG), help="STAR UVT birth/split config to check.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR), help="Directory for summary.json and summary.md.")
    parser.add_argument("--expected-steps", type=int, default=50, help="Expected train.steps for this gate.")
    parser.add_argument(
        "--require-feature-cache",
        action="store_true",
        help="Treat a missing feature cache directory/files as a blocking failure.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write the report and exit 0 even if required artifacts are missing.",
    )
    args = parser.parse_args(argv)

    payload = run_report(
        config_path=args.config,
        output_dir=args.output_dir,
        expected_steps=args.expected_steps,
        require_feature_cache=args.require_feature_cache,
    )
    print(f"status={payload['status']} summary={payload['summary_md']}")
    if payload["status"] != "ready" and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
