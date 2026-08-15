"""Plan or build all six mapped target caches and three G4 capabilities.

The default command is a no-decode, no-write plan.  Conversion requires both
``--execute`` and ``--allow-cache-conversion`` and proceeds one split at a
time, so no scene-wide decoded video or multi-cache working set is resident.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "train"
LANE2 = ROOT / "research_experiments" / "world_foam_lane2"
for import_root in (TRAIN, LANE2, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_worldfoam_public_quality_dataset_capability import (  # noqa: E402
    build_dataset_capability,
)
from config_utils import load_config_file  # noqa: E402
from paper_training_protocol import resolve_paper_training_protocol  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "src"
    / "train_configs"
    / "paper_protocols"
    / "worldfoam_native4d_g4_public_quality_v1.jsonc"
)
HEIGHT = 384
WIDTH = 512
FRAME_COUNT = 300
PAYLOAD_BYTES_PER_VIEW = HEIGHT * WIDTH * FRAME_COUNT * 3


def _repo_path(value: str | Path, *, name: str, must_exist: bool = False) -> Path:
    candidate = Path(value).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{name} leaves the repository") from error
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    return path


def _display(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _scene_contracts(config_path: Path) -> tuple[dict[str, Any], ...]:
    config = load_config_file(_repo_path(config_path, name="G4 config", must_exist=True))
    scenes = config.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 3:
        raise ValueError("G4 cache builder requires the frozen three-scene matrix")
    result: list[dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            raise TypeError("G4 scene contract must be a mapping")
        protocol_path = _repo_path(
            str(scene["protocol"]),
            name="scene protocol",
            must_exist=True,
        )
        protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
        if (
            protocol.dataset.frame_count != FRAME_COUNT
            or protocol.final_stage.image_size.as_list() != [HEIGHT, WIDTH]
        ):
            raise ValueError("G4 cache scene left the frozen 300-frame 384x512 grid")
        root = (
            ROOT
            / "outputs"
            / "cache"
            / "worldfoam_public_quality"
            / protocol.dataset.sample_id
        )
        result.append(
            {
                "scene": str(scene["scene"]),
                "protocol_path": protocol_path,
                "protocol": protocol,
                "cache_root": root,
                "capability_path": root / "public_train_heldout_capability.json",
            }
        )
    return tuple(result)


def build_all_cache_plan(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    from neural3d_mapped_rgb8_adapter import (
        neural3d_mapped_rgb8_offline_preflight,
    )

    scenes = _scene_contracts(config_path)
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    decoder_preflight = neural3d_mapped_rgb8_offline_preflight()
    blockers.extend(
        f"decoder_runtime_unavailable:{blocker}"
        for blocker in decoder_preflight.get("blockers", ())
    )
    for scene in scenes:
        protocol = scene["protocol"]
        for split, cameras in (
            ("train", protocol.dataset.train_cameras),
            ("heldout", protocol.dataset.heldout_cameras),
        ):
            output = scene["cache_root"] / split
            binding = output / "target_dataset_binding.json"
            outputs = (
                binding,
                output / "mapped_rgb8_manifest.json",
                output / "neural3d_conversion_descriptor.json",
                *(output / f"{camera}.rgb8" for camera in cameras),
            )
            existing = [_display(path) for path in outputs if path.exists()]
            if existing:
                blockers.append(
                    f"cache_output_already_exists:{scene['scene']}:{split}"
                )
            rows.append(
                {
                    "scene": scene["scene"],
                    "sample_id": protocol.dataset.sample_id,
                    "target_split": split,
                    "camera_ids": list(cameras),
                    "output_directory": _display(output),
                    "binding": _display(binding),
                    "payload_bytes_per_view": PAYLOAD_BYTES_PER_VIEW,
                    "total_payload_bytes": PAYLOAD_BYTES_PER_VIEW * len(cameras),
                    "existing_outputs": existing,
                }
            )
        if scene["capability_path"].exists():
            blockers.append(f"capability_output_already_exists:{scene['scene']}")
    return {
        "schema_version": 1,
        "kind": "worldfoam-g4-all-public-cache-plan-v1",
        "g4_config": _display(_repo_path(config_path, name="G4 config")),
        "scene_count": len(scenes),
        "cache_count": len(rows),
        "capability_count": len(scenes),
        "stored_frame_indices": [0, FRAME_COUNT - 1],
        "stored_frame_count": FRAME_COUNT,
        "image_size": [HEIGHT, WIDTH],
        "exact_total_payload_bytes": sum(row["total_payload_bytes"] for row in rows),
        "sequential_split_conversion": True,
        "whole_video_materialized": False,
        "decode_started": False,
        "write_started": False,
        "decoder_preflight": decoder_preflight,
        "ready_for_clean_execute": not blockers,
        "blockers": sorted(blockers),
        "caches": rows,
        "capabilities": [
            {
                "scene": scene["scene"],
                "output": _display(scene["capability_path"]),
                "train_binding": _display(
                    scene["cache_root"] / "train" / "target_dataset_binding.json"
                ),
                "heldout_binding": _display(
                    scene["cache_root"] / "heldout" / "target_dataset_binding.json"
                ),
            }
            for scene in scenes
        ],
    }


def _conversion_limits() -> Any:
    from build_worldfoam_mapped_rgb8_cache import (
        WorldFoamMappedRgb8ConversionLimits,
    )

    return WorldFoamMappedRgb8ConversionLimits(
        maximum_raw_dataset_manifest_bytes=1024 * 1024,
        maximum_raw_input_bytes_per_view=2 * 1024**3,
        maximum_total_raw_input_verification_bytes=8 * 1024**3,
        maximum_total_decode_input_bytes=8 * 1024**3,
        maximum_decoded_frame_bytes=1024 * 1024,
        maximum_decode_hash_scratch_bytes=4 * 1024**2,
        maximum_payload_bytes_per_view=200 * 1024**2,
        maximum_total_payload_bytes=400 * 1024**2,
        maximum_transpose_scratch_bytes=64 * 1024**2,
        maximum_temporary_bytes_per_view=400 * 1024**2,
        maximum_total_output_and_temporary_bytes=600 * 1024**2,
        maximum_total_cache_verification_bytes=2 * 1024**3,
        maximum_mapped_manifest_bytes=1024 * 1024,
        maximum_binding_bytes=1024 * 1024,
    )


def _adapter_limits() -> Any:
    from neural3d_mapped_rgb8_adapter import Neural3dMappedRgb8AdapterLimits

    return Neural3dMappedRgb8AdapterLimits(
        maximum_dataset_manifest_bytes=1024 * 1024,
        maximum_poses_bounds_bytes=16 * 1024**2,
        maximum_adapter_source_bytes=4 * 1024**2,
        maximum_total_source_verification_bytes=64 * 1024**2,
        maximum_camera_tensor_bytes=4 * 1024**2,
        maximum_descriptor_bytes=1024 * 1024,
        maximum_mp4_header_read_bytes_per_view=4 * 1024**2,
        maximum_total_mp4_header_read_bytes=16 * 1024**2,
        maximum_native_frame_bytes=32 * 1024**2,
        maximum_python_rgb_scratch_bytes=64 * 1024**2,
        maximum_decoded_native_frames_per_view=1200,
    )


def build_all_caches(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    from neural3d_mapped_rgb8_adapter import (
        build_neural3d_mapped_rgb8_cache,
        neural3d_mapped_rgb8_offline_preflight,
    )

    plan = build_all_cache_plan(config_path)
    if plan["blockers"]:
        raise RuntimeError(
            "G4 cache conversion requires clean absent outputs: "
            + ", ".join(plan["blockers"])
        )
    preflight = neural3d_mapped_rgb8_offline_preflight()
    if preflight.get("ready") is not True:
        raise RuntimeError(
            "G4 cache decoder runtime is unavailable: "
            + ", ".join(preflight.get("blockers", ()))
        )
    scenes = _scene_contracts(config_path)
    cache_receipts: list[dict[str, Any]] = []
    capability_receipts: list[dict[str, Any]] = []
    for scene in scenes:
        protocol = scene["protocol"]
        bindings: dict[str, Path] = {}
        for split in ("train", "heldout"):
            output = scene["cache_root"] / split
            receipt = build_neural3d_mapped_rgb8_cache(
                repository_root=ROOT,
                dataset_manifest_path=_repo_path(
                    protocol.dataset.manifest,
                    name="paper dataset manifest",
                    must_exist=True,
                ),
                dataset_manifest_path_label=protocol.dataset.manifest,
                sample_id=protocol.dataset.sample_id,
                output_directory=output,
                height=HEIGHT,
                width=WIDTH,
                stored_frame_indices=tuple(range(FRAME_COUNT)),
                required_frame_counts=(FRAME_COUNT,),
                conversion_limits=_conversion_limits(),
                adapter_limits=_adapter_limits(),
                target_split=split,
                translation_scale=1.0,
            )
            bindings[split] = receipt.cache.binding_path
            cache_receipts.append(
                {
                    "scene": scene["scene"],
                    "target_split": split,
                    "binding": _display(receipt.cache.binding_path),
                    "binding_sha256": receipt.cache.binding_sha256,
                    "view_ids": list(receipt.view_ids),
                    "exact_total_payload_bytes": (
                        receipt.cache.exact_total_payload_bytes
                    ),
                    "raw_cache_decoded_f32_equality_recomputed": (
                        receipt.cache.raw_cache_decoded_f32_equality_recomputed
                    ),
                }
            )
        capability_path = build_dataset_capability(
            config_path=_repo_path(config_path, name="G4 config", must_exist=True),
            protocol_path=scene["protocol_path"],
            scene=scene["scene"],
            train_binding_path=bindings["train"],
            heldout_binding_path=bindings["heldout"],
            output_path=scene["capability_path"],
        )
        capability_receipts.append(
            {"scene": scene["scene"], "path": _display(capability_path)}
        )
    return {
        "status": "built",
        "cache_count": len(cache_receipts),
        "capability_count": len(capability_receipts),
        "whole_video_materialized": False,
        "sequential_split_conversion": True,
        "caches": cache_receipts,
        "capabilities": capability_receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g4-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-cache-conversion", action="store_true")
    args = parser.parse_args()
    if args.execute != args.allow_cache_conversion:
        raise SystemExit(
            "cache conversion requires both --execute and --allow-cache-conversion"
        )
    result = (
        build_all_caches(args.g4_config)
        if args.execute
        else build_all_cache_plan(args.g4_config)
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
