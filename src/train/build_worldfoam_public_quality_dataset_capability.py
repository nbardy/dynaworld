"""Build the split-safe mapped-cache capability consumed by G4 row workers.

Planning performs no Torch import, cache decode, or accelerator allocation.
``--execute`` rehashes the already-built train and heldout mapped caches and
publishes one canonical capability only after the strict pair verifier passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from config_utils import load_config_file, serialize_config_value
from worldfoam_native4d_public_quality_row import (
    DATASET_CAPABILITY_SCHEMA_VERSION,
    ROOT,
    RowRequest,
    default_dataset_capability_path,
    load_dataset_capability,
    resolve_row_request,
)


PROVIDER_MODULE = "worldfoam_public_quality_dataset_provider"
PROVIDER_CALLABLE = "create_public_quality_dataset"
PROVIDER_SOURCE = ROOT / "src" / "train" / f"{PROVIDER_MODULE}.py"


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        serialize_config_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _atomic_json(path: Path, payload: Any) -> None:
    destination = _repo_path(path, name="dataset capability output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _request_context(
    *,
    config_path: Path,
    protocol_path: Path,
    scene: str,
) -> tuple[RowRequest, tuple[Any, ...]]:
    config = load_config_file(_repo_path(config_path, name="G4 config", must_exist=True))
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("G4 capability builder found no frozen seeds")
    output = (
        _repo_path(str(config["output_root"]), name="G4 output root")
        / str(scene)
        / f"seed_{int(seeds[0])}"
        / "worldfoam_native4d"
        / "g4_row.json"
    )
    request = RowRequest(
        config_path=config_path,
        protocol_path=protocol_path,
        scene=str(scene),
        seed=int(seeds[0]),
        route="worldfoam_native4d",
        output_path=output,
        allow_local_mps_execution=False,
    )
    return request, resolve_row_request(request)


def build_capability_plan(
    *,
    config_path: Path,
    protocol_path: Path,
    scene: str,
    train_binding_path: Path,
    heldout_binding_path: Path,
    output_path: Path | None,
) -> dict[str, Any]:
    request, resolved = _request_context(
        config_path=config_path,
        protocol_path=protocol_path,
        scene=scene,
    )
    protocol = resolved[2]
    destination = output_path or default_dataset_capability_path(protocol)
    train = _repo_path(train_binding_path, name="train binding")
    heldout = _repo_path(heldout_binding_path, name="heldout binding")
    blockers: list[str] = []
    for split, path in (("train", train), ("heldout", heldout)):
        if not path.is_file():
            blockers.append(f"{split}_binding_missing:{_display(path)}")
    if train == heldout:
        blockers.append("train_heldout_binding_paths_identical")
    if not PROVIDER_SOURCE.is_file():
        blockers.append("public_dataset_provider_source_missing")
    payload = {
        "schema_version": 1,
        "kind": "worldfoam-public-train-heldout-cache-capability-plan-v1",
        "scene": scene,
        "sample_id": protocol.dataset.sample_id,
        "frame_count": protocol.dataset.frame_count,
        "image_size": protocol.final_stage.image_size.as_list(),
        "train_cameras": list(protocol.dataset.train_cameras),
        "heldout_cameras": list(protocol.dataset.heldout_cameras),
        "train_binding": _display(train),
        "heldout_binding": _display(heldout),
        "output": _display(_repo_path(destination, name="dataset capability output")),
        "pair_verifier_will_rehash_cache_payloads": True,
        "cache_decode_will_run": False,
        "allocation_started": False,
        "ready_to_execute": not blockers,
        "blockers": sorted(blockers),
        "request_route_used_for_contract_resolution": request.route,
    }
    return {**payload, "plan_sha256": _sha256_json(payload)}


def build_dataset_capability(
    *,
    config_path: Path,
    protocol_path: Path,
    scene: str,
    train_binding_path: Path,
    heldout_binding_path: Path,
    output_path: Path | None = None,
) -> Path:
    from worldfoam_target_dataset_binding import (
        load_target_dataset_binding,
        verify_train_heldout_target_dataset_pair,
    )

    request, resolved = _request_context(
        config_path=config_path,
        protocol_path=protocol_path,
        scene=scene,
    )
    _config, _config_receipt, protocol, _route, scene_receipt, _work = resolved
    train_path = _repo_path(train_binding_path, name="train binding", must_exist=True)
    heldout_path = _repo_path(
        heldout_binding_path,
        name="heldout binding",
        must_exist=True,
    )
    pair = verify_train_heldout_target_dataset_pair(
        train_binding_path=train_path,
        heldout_binding_path=heldout_path,
        required_frame_counts=(protocol.dataset.frame_count,),
    )
    train = load_target_dataset_binding(
        train_path,
        required_frame_counts=(protocol.dataset.frame_count,),
    )
    heldout = load_target_dataset_binding(
        heldout_path,
        required_frame_counts=(protocol.dataset.frame_count,),
    )
    if (
        pair["dataset_id"] != protocol.dataset.sample_id
        or pair["train"]["view_ids"] != list(protocol.dataset.train_cameras)
        or pair["heldout"]["view_ids"] != list(protocol.dataset.heldout_cameras)
    ):
        raise ValueError("verified target pair differs from the frozen protocol")

    def bound_file(path: Path, binding: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": _display(path),
            "bytes": int(path.stat().st_size),
            "sha256": _file_sha256(path),
            "target_split": binding["target_split"],
            "camera_ids": list(binding["camera"]["view_ids"]),
        }

    payload: dict[str, Any] = {
        "schema_version": DATASET_CAPABILITY_SCHEMA_VERSION,
        "kind": "worldfoam-public-train-heldout-cache-v1",
        "scene": scene,
        "sample_id": protocol.dataset.sample_id,
        "protocol_sha256": _file_sha256(
            _repo_path(protocol_path, name="paper protocol", must_exist=True)
        ),
        "dataset_manifest_sha256": scene_receipt["manifest_sha256"],
        "frame_count": protocol.dataset.frame_count,
        "image_size": protocol.final_stage.image_size.as_list(),
        "train_cameras": list(protocol.dataset.train_cameras),
        "heldout_cameras": list(protocol.dataset.heldout_cameras),
        "train_binding": bound_file(train_path, train),
        "heldout_binding": bound_file(heldout_path, heldout),
        "provider_factory": {
            "module": PROVIDER_MODULE,
            "callable": PROVIDER_CALLABLE,
            "source_path": _display(PROVIDER_SOURCE),
            "source_sha256": _file_sha256(PROVIDER_SOURCE),
        },
        "public_data": True,
        "calibrated_multiview": True,
        "selected_pixel_reads": True,
        "full_frame_materialization_required": False,
        "initialization_sha256": scene_receipt["initialization_sha256"],
        "compiler_sha256": scene_receipt["compiler_sha256"],
        "worldfoam_runtime_sha256": scene_receipt["worldfoam_runtime_sha256"],
        "capability_sha256": "",
    }
    payload["capability_sha256"] = _sha256_json(
        {key: value for key, value in payload.items() if key != "capability_sha256"}
    )
    destination = _repo_path(
        output_path or default_dataset_capability_path(protocol),
        name="dataset capability output",
    )
    _atomic_json(destination, payload)
    try:
        verified = load_dataset_capability(
            destination,
            request=request,
            protocol=protocol,
            scene_receipt=scene_receipt,
        )
        if verified.get("_verified_pair_receipt") != pair:
            raise ArithmeticError("published capability pair receipt changed")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--g4-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--train-binding", type=Path, required=True)
    parser.add_argument("--heldout-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    kwargs = {
        "config_path": args.g4_config,
        "protocol_path": args.protocol,
        "scene": args.scene,
        "train_binding_path": args.train_binding,
        "heldout_binding_path": args.heldout_binding,
        "output_path": args.output,
    }
    if not args.execute:
        print(json.dumps(build_capability_plan(**kwargs), sort_keys=True, indent=2))
        return
    output = build_dataset_capability(**kwargs)
    print(
        json.dumps(
            {
                "status": "capability_published",
                "output": _display(output),
                "sha256": _file_sha256(output),
                "cache_decode_ran": False,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

