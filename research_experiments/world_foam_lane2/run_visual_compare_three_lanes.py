#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_TRAIN = PROJECT_ROOT / "src" / "train"
if str(SRC_TRAIN) not in sys.path:
    sys.path.insert(0, str(SRC_TRAIN))

from config_utils import load_config_file  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "visual_comparisons" / "three_lane_visual_compare_summary.json"
MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".mp4", ".gif", ".webm"}
MEDIA_DEPS = ("imageio", "moviepy")


@dataclass(frozen=True)
class LaneSpec:
    name: str
    representation: str
    config_path: Path
    note: str


TINY_LANES = (
    LaneSpec(
        name="worldfoam_dynamic_powerfoam_metal",
        representation="worldfoam_dynamic_powerfoam_metal",
        config_path=PROJECT_ROOT
        / "src"
        / "train_configs"
        / "local_mac_dynamic_powerfoam_metal_rbf_color_only_fixed_geometry_video_1024_16f_40step_smoke.jsonc",
        note="Dynamic WorldFoam/PowerFoam Metal trainer smoke with explicit output_dir media.",
    ),
    LaneSpec(
        name="worldtubes_star_uvt_metal",
        representation="worldtubes_star_uvt_metal",
        config_path=PROJECT_ROOT
        / "src"
        / "train_configs"
        / "visual_compare_star_uvt_worldtubes_metal_64_16f_20step.jsonc",
        note="STAR UVT / WorldTubes Metal tile trainer on the same tiny visual-compare clip.",
    ),
    LaneSpec(
        name="dynamic_gsplat_fast_mac_metal",
        representation="dynamic_gsplat_fast_mac_metal",
        config_path=PROJECT_ROOT
        / "src"
        / "train_configs"
        / "visual_compare_dynamic_gsplat_fast_mac_metal_64_16f_20step.jsonc",
        note="Base dynamic 3DGS trainer using the fast-mac Metal rasterizer path.",
    ),
)
MEDIUM_LANES = (
    LaneSpec(
        name="worldfoam_dynamic_powerfoam_metal",
        representation="worldfoam_dynamic_powerfoam_metal",
        config_path=PROJECT_ROOT
        / "src"
        / "train_configs"
        / "visual_compare_worldfoam_dynamic_powerfoam_metal_128_16f_40step.jsonc",
        note="128px Dynamic WorldFoam/PowerFoam Metal trainer visual tier with explicit output_dir media.",
    ),
    LaneSpec(
        name="worldtubes_star_uvt_metal",
        representation="worldtubes_star_uvt_metal",
        config_path=PROJECT_ROOT
        / "src"
        / "train_configs"
        / "visual_compare_star_uvt_worldtubes_metal_128_16f_20step.jsonc",
        note="128px STAR UVT / WorldTubes Metal tile visual tier on the shared local clip.",
    ),
    LaneSpec(
        name="dynamic_gsplat_fast_mac_metal",
        representation="dynamic_gsplat_fast_mac_metal",
        config_path=PROJECT_ROOT
        / "src"
        / "train_configs"
        / "visual_compare_dynamic_gsplat_fast_mac_metal_128_16f_20step.jsonc",
        note="128px base dynamic 3DGS trainer using the fast-mac Metal rasterizer path.",
    ),
)
CAPACITY_LANES = (
    LaneSpec(
        name="worldfoam_dynamic_powerfoam_metal",
        representation="worldfoam_dynamic_powerfoam_metal",
        config_path=PROJECT_ROOT
        / "src"
        / "train_configs"
        / "visual_compare_worldfoam_dynamic_powerfoam_metal_128_16f_80step_2048cells.jsonc",
        note="128px capacity-tier Dynamic WorldFoam/PowerFoam Metal trainer with 2048 cells.",
    ),
    LaneSpec(
        name="worldtubes_star_uvt_metal",
        representation="worldtubes_star_uvt_metal",
        config_path=PROJECT_ROOT
        / "src"
        / "train_configs"
        / "visual_compare_star_uvt_worldtubes_metal_128_16f_60step_2048tubes.jsonc",
        note="128px capacity-tier STAR UVT / WorldTubes Metal tile trainer with 2048 tubes.",
    ),
    LaneSpec(
        name="dynamic_gsplat_fast_mac_metal",
        representation="dynamic_gsplat_fast_mac_metal",
        config_path=PROJECT_ROOT
        / "src"
        / "train_configs"
        / "visual_compare_dynamic_gsplat_fast_mac_metal_128_16f_60step_4096gs.jsonc",
        note="128px capacity-tier base dynamic 3DGS trainer with 4096 fast-mac Metal Gaussians.",
    ),
)
TIER_LANES = {
    "tiny": TINY_LANES,
    "medium": MEDIUM_LANES,
    "capacity": CAPACITY_LANES,
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def lanes_for_tier(tier: str) -> tuple[LaneSpec, ...]:
    try:
        return TIER_LANES[str(tier)]
    except KeyError as exc:
        known = ", ".join(sorted(TIER_LANES))
        raise ValueError(f"Unknown visual comparison tier {tier!r}. Expected one of: {known}.") from exc


def lane_by_name(tier: str = "tiny") -> dict[str, LaneSpec]:
    return {lane.name: lane for lane in lanes_for_tier(tier)}


def known_lane_names() -> set[str]:
    return {lane.name for lanes in TIER_LANES.values() for lane in lanes}


def selected_lanes(names: list[str] | None, *, tier: str = "tiny") -> list[LaneSpec]:
    tier_lanes = lanes_for_tier(tier)
    if not names:
        return list(tier_lanes)
    known = lane_by_name(tier)
    lanes: list[LaneSpec] = []
    for name in names:
        if name not in known:
            raise ValueError(f"Unknown lane {name!r}. Expected one of: {', '.join(sorted(known))}.")
        lanes.append(known[name])
    return lanes


def trainer_command(config_path: Path, *, include_media_deps: bool = True) -> list[str]:
    if include_media_deps:
        command = ["uv", "run"]
        for dep in MEDIA_DEPS:
            command.extend(["--with", dep])
        command.extend(["python", "src/train/train.py", rel(config_path)])
        return command
    return [sys.executable, "src/train/train.py", rel(config_path)]


def trainer_env() -> dict[str, str]:
    env = dict(os.environ)
    py_path = str(SRC_TRAIN)
    if env.get("PYTHONPATH"):
        py_path = py_path + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = py_path
    return env


def config_arch(cfg: dict[str, Any]) -> str:
    return str(cfg.get("arch", "<missing>"))


def config_backend_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    arch = config_arch(cfg)
    render = cfg.get("render", {}) if isinstance(cfg.get("render"), dict) else {}
    data = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    logging = cfg.get("logging", {}) if isinstance(cfg.get("logging"), dict) else {}
    summary: dict[str, Any] = {
        "arch": arch,
        "render_size": render.get("render_size", data.get("target_size")),
        "device": (cfg.get("train") or {}).get("device"),
        "steps": (cfg.get("train") or {}).get("steps"),
        "wandb_mode": logging.get("wandb_mode"),
        "wandb_enabled": logging.get("wandb_enabled"),
    }
    if arch == "star_uvt_video_overfit":
        uvt = cfg.get("uvt", {})
        summary.update(
            {
                "metal_backend": uvt.get("render_backend"),
                "sample_emission_mode": uvt.get("sample_emission_mode"),
                "tube_count": uvt.get("tube_count"),
            }
        )
    elif arch in {"dynamic_powerfoam_metal", "powerfoam_metal"}:
        summary.update(
            {
                "metal_backend": "torch_dynamic_powerfoam_metal"
                if arch == "dynamic_powerfoam_metal"
                else "torch_powerfoam_metal",
                "cell_count": (cfg.get("model") or {}).get("cells"),
            }
        )
    else:
        fast_mac = render.get("fast_mac", {}) if isinstance(render.get("fast_mac"), dict) else {}
        summary.update(
            {
                "renderer": render.get("renderer"),
                "fast_mac_rgb_variant": fast_mac.get("rgb_variant", fast_mac.get("variant")),
                "fast_mac_feature_variant": fast_mac.get("feature_variant"),
            }
        )
    return summary


def declared_artifact_paths(cfg: dict[str, Any]) -> list[tuple[str, Path]]:
    artifacts: list[tuple[str, Path]] = []
    output = cfg.get("output")
    if isinstance(output, dict):
        for key in ("out_json", "contact_sheet", "side_by_side_video"):
            raw = output.get(key)
            if raw:
                artifacts.append((key, project_path(str(raw))))
    logging = cfg.get("logging")
    if isinstance(logging, dict) and logging.get("output_dir"):
        output_dir = project_path(str(logging["output_dir"]))
        step = int((cfg.get("train") or {}).get("steps", 0))
        if step > 0 and bool(logging.get("always_log_last_step", True)):
            artifacts.extend(
                [
                    ("preview_image", output_dir / f"preview_step_{step:04d}.png"),
                    ("render_video", output_dir / f"render_step_{step:04d}.mp4"),
                    ("side_by_side_video", output_dir / f"side_by_side_step_{step:04d}.mp4"),
                ]
            )
    return artifacts


def artifact_status(label: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "label": label,
        "path": rel(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists and path.is_file() else None,
    }


def collect_declared_artifacts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [artifact_status(label, path) for label, path in declared_artifact_paths(cfg)]


def collect_recent_wandb_media(start_time_s: float, *, max_items: int = 32) -> list[dict[str, Any]]:
    wandb_root = PROJECT_ROOT / "wandb"
    if not wandb_root.exists():
        return []
    media: list[dict[str, Any]] = []
    for path in sorted(wandb_root.glob("offline-run-*/files/media/**/*")):
        if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime + 1.0 < start_time_s:
            continue
        media.append(
            {
                "path": rel(path),
                "bytes": stat.st_size,
                "mtime_s": stat.st_mtime,
            }
        )
    return sorted(media, key=lambda item: str(item["path"]))[-max_items:]


def lane_summary_base(lane: LaneSpec, cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    return {
        "name": lane.name,
        "representation": lane.representation,
        "note": lane.note,
        "config": rel(lane.config_path),
        "data": {
            "video_path": data.get("video_path"),
            "manifest_path": data.get("manifest_path"),
            "frame_source": data.get("frame_source"),
            "max_frames": data.get("max_frames"),
        },
        "backend": config_backend_summary(cfg),
        "declared_artifacts": collect_declared_artifacts(cfg),
    }


def run_lane(
    lane: LaneSpec,
    *,
    dry_run: bool,
    logs_dir: Path,
    timeout_s: float,
    include_media_deps: bool,
) -> dict[str, Any]:
    if not lane.config_path.exists():
        return {
            "name": lane.name,
            "representation": lane.representation,
            "status": "missing_config",
            "config": rel(lane.config_path),
        }
    cfg = load_config_file(lane.config_path)
    summary = lane_summary_base(lane, cfg)
    command = trainer_command(lane.config_path, include_media_deps=include_media_deps)
    summary["command"] = command
    if dry_run:
        summary["status"] = "planned"
        return summary

    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{lane.name}.stdout.txt"
    stderr_path = logs_dir / f"{lane.name}.stderr.txt"
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=trainer_env(),
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            exit_code = int(completed.returncode)
            status = "ok" if exit_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            exit_code = None
            status = "timeout"

    summary.update(
        {
            "status": status,
            "exit_code": exit_code,
            "elapsed_s": time.time() - start,
            "stdout": rel(stdout_path),
            "stderr": rel(stderr_path),
            "declared_artifacts": collect_declared_artifacts(cfg),
            "recent_wandb_media": collect_recent_wandb_media(start),
        }
    )
    missing = [
        artifact["path"]
        for artifact in summary["declared_artifacts"]
        if not bool(artifact.get("exists"))
    ]
    if status == "ok" and missing:
        summary["status"] = "missing_declared_artifacts"
        summary["missing_declared_artifacts"] = missing
    return summary


def overall_status(lanes: list[dict[str, Any]]) -> str:
    if all(lane.get("status") == "planned" for lane in lanes):
        return "planned"
    if all(lane.get("status") == "ok" for lane in lanes):
        return "ok"
    if any(lane.get("status") in {"failed", "timeout", "missing_config"} for lane in lanes):
        return "failed"
    if any(lane.get("status") == "missing_declared_artifacts" for lane in lanes):
        return "missing_declared_artifacts"
    return "partial"


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    lanes = []
    logs_dir = project_path(args.logs_dir)
    tier = str(getattr(args, "tier", "tiny"))
    for lane in selected_lanes(args.lane, tier=tier):
        row = run_lane(
            lane,
            dry_run=bool(args.dry_run),
            logs_dir=logs_dir,
            timeout_s=float(args.timeout_s),
            include_media_deps=not bool(args.no_media_deps),
        )
        lanes.append(row)
        if row["status"] in {"failed", "timeout"} and not bool(args.continue_on_failure):
            break
    return {
        "status": overall_status(lanes),
        "dry_run": bool(args.dry_run),
        "tier": tier,
        "created_at_s": time.time(),
        "objective": "visual comparison of WorldFoam, WorldTubes/STAR UVT, and base dynamic 3DGS Metal-backed trainers",
        "lanes": lanes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or plan the three-lane WorldFoam / WorldTubes / dynamic-gsplat visual comparison."
    )
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Summary JSON path.")
    parser.add_argument(
        "--tier",
        default="tiny",
        choices=sorted(TIER_LANES),
        help=(
            "Visual comparison tier. tiny is the 64px smoke; medium is the 128px visual tier; "
            "capacity is a modest 128px capacity/step scale-up."
        ),
    )
    parser.add_argument(
        "--logs-dir",
        default="outputs/visual_comparisons/logs",
        help="Directory for per-lane stdout/stderr captures.",
    )
    parser.add_argument(
        "--lane",
        action="append",
        choices=sorted(known_lane_names()),
        help="Lane to run. Repeat to run a subset; default runs all lanes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the command/artifact plan without training.")
    parser.add_argument("--timeout-s", type=float, default=1800.0, help="Per-lane subprocess timeout.")
    parser.add_argument(
        "--no-media-deps",
        action="store_true",
        help="Do not launch trainer subprocesses with uv --with imageio --with moviepy.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Keep running later lanes after a failed or timed-out lane.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    out = project_path(args.out)
    write_json(out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
