"""Production public-data provider for the frozen WorldFoam G4 rows.

The provider is instantiated only after the row worker's allocation-free
preflight has verified both mapped-RGB8 cache bindings.  It keeps no decoded
video resident: every target request opens the split-specific pixel-time cache
for one bounded selected-pixel read, and every ray request constructs only the
requested calibrated rays on CPU.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
_ATTESTATION = {
    "public_data": True,
    "calibrated_multiview": True,
    "procedural_target": False,
    "train_cache_bound": True,
    "heldout_cache_bound": True,
    "selected_pixel_reads": True,
    "full_frame_materialization_required": False,
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _repo_file(value: str | Path, *, name: str) -> Path:
    candidate = Path(value).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{name} leaves the repository") from error
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    return path


def _bound_path(capability: Mapping[str, Any], split: str) -> Path:
    receipt = capability[f"{split}_binding"]
    if not isinstance(receipt, Mapping):
        raise TypeError(f"{split} binding receipt must be a mapping")
    return _repo_file(str(receipt["path"]), name=f"{split} target binding")


def _cache_manifest_path(binding_path: Path, binding: Mapping[str, Any]) -> Path:
    label = binding["cache"]["manifest"]["path_label"]
    path = (binding_path.parent / str(label)).resolve()
    try:
        path.relative_to(binding_path.parent.resolve())
    except ValueError as error:
        raise ValueError("mapped target manifest escaped its binding directory") from error
    if not path.is_file():
        raise FileNotFoundError(f"mapped target manifest is missing: {path}")
    return path


def _logical_frames(binding: Mapping[str, Any], frame_count: int) -> tuple[int, ...]:
    matches = [
        record
        for record in binding["logical_frame_maps"]
        if record.get("frame_count") == int(frame_count)
    ]
    if len(matches) != 1:
        raise ValueError("target binding has no unique logical frame map")
    return tuple(int(value) for value in matches[0]["source_frame_indices"])


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest()


def _seed_tensor_content_sha256(name: str, tensor: Any) -> str:
    """Content-bind one seed tensor without recording a host-specific path."""

    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "name": str(name),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "bytes": int(value.numel() * value.element_size()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _representation_seed_content_sha256(seed: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "schema_version": 1,
            "source_time_provenance": str(seed["source_time_provenance"]),
            "tensor_sha256": {
                key: _seed_tensor_content_sha256(key, seed[key])
                for key in (
                    "positions0_f32_cpu",
                    "colors_f32_cpu",
                    "source_frame_indices_i64_cpu",
                )
            },
        }
    )


def _assert_tensor_identity(tensor: Any, identity: Mapping[str, Any], *, name: str) -> None:
    import torch

    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    expected_shape = tuple(int(item) for item in identity["shape"])
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != expected_shape
        or int(value.numel() * value.element_size()) != int(identity["bytes"])
        or _tensor_sha256(value) != identity["sha256"]
    ):
        raise ValueError(f"calibrated {name} differs from the target-cache binding")


def _manifest_record(protocol: Any) -> dict[str, Any]:
    path = _repo_file(protocol.dataset.manifest, name="paper dataset manifest")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict) and value.get("sample_id") == protocol.dataset.sample_id:
            records.append(value)
    if len(records) != 1:
        raise ValueError("paper manifest has no unique public sample record")
    return records[0]


def _clone_camera(camera: Any) -> Any:
    import torch
    from camera import CameraSpec

    def scalar(value: Any) -> float:
        return float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)

    distortion = camera.distortion
    if distortion is not None:
        distortion = torch.as_tensor(
            distortion,
            dtype=torch.float32,
            device="cpu",
        ).clone().contiguous()
    return CameraSpec(
        fx=scalar(camera.fx),
        fy=scalar(camera.fy),
        cx=scalar(camera.cx),
        cy=scalar(camera.cy),
        camera_to_world=camera.camera_to_world.detach()
        .to(device="cpu", dtype=torch.float32)
        .clone()
        .contiguous(),
        lens_model=camera.lens_model,
        distortion=distortion,
    )


class WorldFoamPublicQualityDataset:
    """One owner-bound public dataset with mapped targets and calibrated rays."""

    def __init__(
        self,
        *,
        context: Any,
        capability: Mapping[str, Any],
        train_target_provider: Any,
        train_ray_provider: Any,
        heldout_target_provider: Any,
        heldout_ray_provider: Any,
        frame_times: tuple[float, ...],
        sealed_inputs_factory: Any,
        seed: Mapping[str, Any],
        maximum_source_decode_tensor_bytes: int,
    ) -> None:
        self.sample_id = str(context.protocol.dataset.sample_id)
        self.train_cameras = tuple(context.protocol.dataset.train_cameras)
        self.heldout_cameras = tuple(context.protocol.dataset.heldout_cameras)
        self.frame_count = int(context.protocol.dataset.frame_count)
        self.height = int(context.protocol.final_stage.image_size.height)
        self.width = int(context.protocol.final_stage.image_size.width)
        self._context = context
        self._capability = capability
        self._train_target_provider = train_target_provider
        self._train_ray_provider = train_ray_provider
        self._heldout_target_provider = heldout_target_provider
        self._heldout_ray_provider = heldout_ray_provider
        self._frame_times = frame_times
        self._sealed_inputs_factory = sealed_inputs_factory
        self._sealed_inputs: Any | None = None
        self._seed = dict(seed)
        self._maximum_source_decode_tensor_bytes = int(
            maximum_source_decode_tensor_bytes
        )
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("public-quality dataset is closed")

    def attestation(self) -> Mapping[str, Any]:
        self._require_open()
        return dict(_ATTESTATION)

    def _read(self, request: Any, *, split: str) -> Any:
        import torch
        from camera import build_camera_rays_at_pixels
        from worldfoam_native4d_public_quality_row import PixelChunkPayload

        self._require_open()
        explicit_pixels = getattr(request, "pixel_ids", None)
        if (
            request.split != split
            or request.image_height != self.height
            or request.image_width != self.width
            or request.pixel_count < 1
            or request.pixel_start < 0
            or (
                explicit_pixels is None
                and request.pixel_stop > self.height * self.width
            )
        ):
            raise ValueError("public selected-pixel request left its frozen grid")
        if explicit_pixels is not None:
            explicit_pixels = tuple(int(value) for value in explicit_pixels)
            if (
                len(explicit_pixels) != int(request.pixel_count)
                or tuple(sorted(explicit_pixels)) != explicit_pixels
                or len(set(explicit_pixels)) != len(explicit_pixels)
                or explicit_pixels[0] < 0
                or explicit_pixels[-1] >= self.height * self.width
            ):
                raise ValueError("explicit sensor-pixel ids violate the selected-ray contract")
        target_provider = (
            self._train_target_provider
            if split == "train"
            else self._heldout_target_provider
        )
        ray_provider = (
            self._train_ray_provider if split == "train" else self._heldout_ray_provider
        )
        if not 0 <= request.camera_index < target_provider.view_count:
            raise IndexError("public request camera index is out of range")
        if not 0 <= request.frame_index < self.frame_count:
            raise IndexError("public request frame index is out of range")
        pixels = (
            tuple(range(int(request.pixel_start), int(request.pixel_stop)))
            if explicit_pixels is None
            else explicit_pixels
        )
        read = target_provider.select_view_frame_pixels_cpu(
            (int(request.camera_index),) * request.pixel_count,
            (int(request.frame_index),) * request.pixel_count,
            pixels,
            maximum_source_decode_tensor_bytes=(
                self._maximum_source_decode_tensor_bytes
            ),
        )
        pixel_tensor = (
            torch.arange(
                request.pixel_start,
                request.pixel_stop,
                dtype=torch.long,
                device="cpu",
            )
            if explicit_pixels is None
            else torch.tensor(explicit_pixels, dtype=torch.long, device="cpu")
        )
        origins, directions = build_camera_rays_at_pixels(
            ray_provider.cameras[request.camera_index][request.frame_index],
            pixel_tensor,
            height=self.height,
            width=self.width,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        rays = torch.cat((origins, directions), dim=-1).contiguous()
        selected_read_receipt: Any = read
        if explicit_pixels is not None:
            selected_read_receipt = {
                **dict(read.__dict__),
                "requested_pixel_ids_sha256": _canonical_sha256(explicit_pixels),
            }
        return PixelChunkPayload(
            target_rgb_f32_cpu=read.rgb_f32_cpu,
            rays_f32_cpu=rays,
            selected_read_receipt=selected_read_receipt,
        )

    def read_train_chunk(self, request: Any) -> Any:
        return self._read(request, split="train")

    def read_heldout_chunk(self, request: Any) -> Any:
        return self._read(request, split="heldout")

    def camera_spec(
        self,
        *,
        split: str,
        camera_index: int,
        frame_index: int,
    ) -> Any:
        self._require_open()
        provider = {
            "train": self._train_ray_provider,
            "heldout": self._heldout_ray_provider,
        }.get(split)
        if provider is None:
            raise ValueError("camera split must be train or heldout")
        if not 0 <= int(camera_index) < provider.view_count:
            raise IndexError("camera index is out of range")
        if not 0 <= int(frame_index) < provider.frame_count:
            raise IndexError("camera frame index is out of range")
        return _clone_camera(provider.cameras[int(camera_index)][int(frame_index)])

    def worldfoam_training_inputs(self) -> Mapping[str, Any]:
        self._require_open()
        if self._sealed_inputs is None:
            self._sealed_inputs = self._sealed_inputs_factory(self)
        self._sealed_inputs.assert_current(dataset=self, context=self._context)
        return self._sealed_inputs

    def representation_seed(self) -> Mapping[str, Any]:
        import torch

        self._require_open()
        positions = self._seed["positions0_f32_cpu"].clone().contiguous()
        colors = self._seed["colors_f32_cpu"].clone().contiguous()
        source_frames = self._seed["source_frame_indices_i64_cpu"].clone().contiguous()
        if positions.data_ptr() == self._seed["positions0_f32_cpu"].data_ptr():
            raise RuntimeError("representation seed clone unexpectedly aliases its template")
        if colors.data_ptr() == self._seed["colors_f32_cpu"].data_ptr():
            raise RuntimeError("representation seed clone unexpectedly aliases its template")
        if source_frames.data_ptr() == self._seed[
            "source_frame_indices_i64_cpu"
        ].data_ptr():
            raise RuntimeError("representation source-time clone unexpectedly aliases its template")
        if positions.dtype != torch.float32 or colors.dtype != torch.float32:
            raise RuntimeError("representation seed template dtype changed")
        if source_frames.dtype != torch.int64:
            raise RuntimeError("representation source-time template dtype changed")
        return {
            "positions0_f32_cpu": positions,
            "colors_f32_cpu": colors,
            "source_frame_indices_i64_cpu": source_frames,
            "source_time_provenance": self._seed["source_time_provenance"],
            "initializer_generation_digest": self._seed[
                "initializer_generation_digest"
            ],
            "material_seed_generation_digest": self._seed[
                "material_seed_generation_digest"
            ],
            "sites_content_digest": self._seed["sites_content_digest"],
            "material_content_digest": self._seed["material_content_digest"],
            "representation_seed_content_sha256": self._seed[
                "representation_seed_content_sha256"
            ],
        }

    def close(self) -> None:
        self._closed = True
        self._sealed_inputs = None


def create_public_quality_dataset(
    *,
    context: Any,
    capability: Mapping[str, Any],
) -> WorldFoamPublicQualityDataset:
    """Build the exact public provider named by a verified capability receipt."""

    import torch
    from multicam_video_data import (
        cameras_from_K_w2c,
        heldout_cameras_from_K_w2c,
        load_multicam_video_bundle,
    )
    from paper_kinetic_active_track_program_factory import (
        PaperKineticActiveP0TrackProgramFactoryConfig,
        prepare_paper_kinetic_active_p0_track_program_factory,
    )
    from paper_kinetic_fixed_camera_combined_state import (
        PaperKineticFixedCameraCombinedSGDPolicy,
    )
    from paper_kinetic_fixed_site_material_state import (
        PaperKineticFixedSiteMaterialParameterization,
        PaperKineticFixedSiteMaterialSGDPolicy,
    )
    from paper_kinetic_lazy_full_geometry_step import (
        PaperKineticLazyFullGeometryMemoryPolicy,
    )
    from paper_kinetic_lazy_program_bundles import (
        prepare_paper_kinetic_lazy_program_bundle_provider,
    )
    from paper_kinetic_world_initializer import (
        prepare_paper_kinetic_point_cloud_world_initializer,
    )
    from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path
    from powerfoam_training_data import (
        MappedRgb8PowerFoamTargetSource,
        PowerFoamRayProvider,
        PowerFoamTargetProvider,
    )
    from worldfoam_public_quality_inputs import (
        seal_worldfoam_public_training_inputs,
    )
    from worldfoam_target_dataset_binding import load_target_dataset_binding

    ensure_worldfoam_lane2_research_path()
    from kinetic_compiled_cpu_artifact_store import (
        PaperKineticCompiledCpuArtifactStorePolicy,
    )
    from kinetic_lazy_native_material_step import PaperKineticLazyNativeMemoryPolicy

    if capability is not context.dataset_capability:
        raise ValueError("dataset capability object changed after row preflight")
    pair_receipt = capability.get("_verified_pair_receipt")
    if not isinstance(pair_receipt, Mapping):
        raise ValueError("dataset capability lacks its verified split-pair receipt")
    frame_count = int(context.protocol.dataset.frame_count)
    height = int(context.protocol.final_stage.image_size.height)
    width = int(context.protocol.final_stage.image_size.width)
    train_binding_path = _bound_path(capability, "train")
    heldout_binding_path = _bound_path(capability, "heldout")
    train_binding = load_target_dataset_binding(
        train_binding_path,
        required_frame_counts=(frame_count,),
        verify_cache_files=False,
    )
    heldout_binding = load_target_dataset_binding(
        heldout_binding_path,
        required_frame_counts=(frame_count,),
        verify_cache_files=False,
    )

    def mapped_provider(
        binding_path: Path,
        binding: Mapping[str, Any],
        camera_ids: tuple[str, ...],
    ) -> PowerFoamTargetProvider:
        payload_bytes = tuple(
            int(record["payload"]["size_bytes"])
            for record in binding["cache"]["views"]
        )
        source = MappedRgb8PowerFoamTargetSource.from_manifest(
            _cache_manifest_path(binding_path, binding),
            maximum_mapped_payload_bytes=max(payload_bytes),
            maximum_total_payload_verification_bytes=sum(payload_bytes),
            expected_view_ids=camera_ids,
            logical_frame_indices=_logical_frames(binding, frame_count),
            full_frame_source=None,
        )
        return PowerFoamTargetProvider(source=source, device=torch.device("cpu"))

    train_target_provider = mapped_provider(
        train_binding_path,
        train_binding,
        tuple(context.protocol.dataset.train_cameras),
    )
    heldout_target_provider = mapped_provider(
        heldout_binding_path,
        heldout_binding,
        tuple(context.protocol.dataset.heldout_cameras),
    )

    record = _manifest_record(context.protocol)
    data_cfg = {
        "frame_source": "multicam_val",
        "max_frames": frame_count,
        "multicam_manifest": str(
            _repo_file(context.protocol.dataset.manifest, name="paper dataset manifest")
        ),
        "multicam_split": str(record["split"]),
        "multicam_sample_id": context.protocol.dataset.sample_id,
        "multicam_train_cameras": list(context.protocol.dataset.train_cameras),
        "multicam_heldout_cameras": list(context.protocol.dataset.heldout_cameras),
        "multicam_anchor_camera": str(record["anchor_camera"]),
        "multicam_condition_camera": str(record["condition_camera"]),
    }
    bundle = load_multicam_video_bundle(
        data_cfg=data_cfg,
        camera_cfg={
            "lens_model": "pinhole",
            "rig_init": "neural_3d_video",
            "n3d_translation_scale": 1.0,
        },
        target_size=(height, width),
        device=torch.device("cpu"),
        frame_device=torch.device("cpu"),
        defer_video_frames=True,
    )
    if not bundle.deferred_target_frames:
        raise MemoryError("public camera load decoded target frames")
    if tuple(bundle.train_camera_names) != tuple(context.protocol.dataset.train_cameras):
        raise ValueError("loaded train camera order differs from the protocol")
    if tuple(bundle.heldout_camera_names or ()) != tuple(
        context.protocol.dataset.heldout_cameras
    ):
        raise ValueError("loaded heldout camera order differs from the protocol")
    if bundle.heldout_K is None or bundle.heldout_w2c is None:
        raise ValueError("public camera bundle lacks heldout calibration")
    frame_times_tensor = bundle.condition_sequence.frame_times
    _assert_tensor_identity(
        frame_times_tensor,
        train_binding["camera"]["frame_times"],
        name="frame times",
    )
    _assert_tensor_identity(
        bundle.train_K,
        train_binding["camera"]["K"],
        name="train intrinsics",
    )
    _assert_tensor_identity(
        bundle.train_w2c,
        train_binding["camera"]["w2c"],
        name="train poses",
    )
    _assert_tensor_identity(
        bundle.heldout_K,
        heldout_binding["camera"]["K"],
        name="heldout intrinsics",
    )
    _assert_tensor_identity(
        bundle.heldout_w2c,
        heldout_binding["camera"]["w2c"],
        name="heldout poses",
    )
    train_cameras = cameras_from_K_w2c(
        bundle.train_K,
        bundle.train_w2c,
        lens_models=bundle.train_lens_models,
        distortions=bundle.train_distortions,
    )
    heldout_cameras = heldout_cameras_from_K_w2c(
        bundle.heldout_K,
        bundle.heldout_w2c,
        lens_models=bundle.heldout_lens_models,
        distortions=bundle.heldout_distortions,
    )
    train_ray_provider = PowerFoamRayProvider(
        cameras=train_cameras,
        height=height,
        width=width,
        device=torch.device("cpu"),
    )
    heldout_ray_provider = PowerFoamRayProvider(
        cameras=heldout_cameras,
        height=height,
        width=width,
        device=torch.device("cpu"),
    )
    frame_times = tuple(float(value) for value in frame_times_tensor.reshape(-1).tolist())
    initialization = dict(context.scene_receipt["initialization"]["initializer"])
    initialization["source_path"] = str(
        _repo_file(initialization["source_path"], name="point-cloud initializer asset")
    )
    world_initializer = prepare_paper_kinetic_point_cloud_world_initializer(
        initialization
    )
    compiler = context.scene_receipt["compiler"]
    program_factory = prepare_paper_kinetic_active_p0_track_program_factory(
        PaperKineticActiveP0TrackProgramFactoryConfig(**dict(compiler))
    )
    runtime = context.scene_receipt["worldfoam_runtime"]
    artifact_store_policy = PaperKineticCompiledCpuArtifactStorePolicy(
        **dict(runtime["artifact_store_policy"])
    )
    lazy_memory_policy = PaperKineticLazyNativeMemoryPolicy(
        **dict(runtime["lazy_memory_policy"])
    )
    full_geometry_memory_policy = PaperKineticLazyFullGeometryMemoryPolicy(
        **dict(runtime["full_geometry_memory_policy"])
    )
    combined_sgd_policy = PaperKineticFixedCameraCombinedSGDPolicy(
        **dict(runtime["combined_sgd_policy"])
    )
    material_parameterization = PaperKineticFixedSiteMaterialParameterization(
        **dict(runtime["material_parameterization"])
    )
    material_sgd_policy = PaperKineticFixedSiteMaterialSGDPolicy(
        **dict(runtime["material_sgd_policy"])
    )
    dataset_generation_digest = _canonical_sha256(
        {
            "split": "train",
            "pair_receipt_sha256": pair_receipt["pair_receipt_sha256"],
            "binding_sha256": train_binding["binding_sha256"],
            "camera_generation_digest": train_binding["camera"][
                "camera_generation_digest"
            ],
            "capability_sha256": capability["capability_sha256"],
        }
    )
    heldout_dataset_generation_digest = _canonical_sha256(
        {
            "split": "heldout",
            "pair_receipt_sha256": pair_receipt["pair_receipt_sha256"],
            "binding_sha256": heldout_binding["binding_sha256"],
            "camera_generation_digest": heldout_binding["camera"][
                "camera_generation_digest"
            ],
            "capability_sha256": capability["capability_sha256"],
        }
    )
    seed_provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=dataset_generation_digest,
        target_provider=train_target_provider,
        ray_provider=train_ray_provider,
        frame_times=frame_times,
        height=height,
        width=width,
        maximum_tracks_per_bundle=int(runtime["maximum_tracks_per_bundle"]),
        maximum_observations_per_bundle=int(
            runtime["maximum_observations_per_bundle"]
        ),
        maximum_rows_per_native_block=int(runtime["maximum_rows_per_native_block"]),
        world_initializer=world_initializer,
        program_factory=program_factory,
    )
    material_seed = world_initializer.initialize_p0_material(seed_provider.world.sites)
    source_frame_indices = torch.full(
        (int(seed_provider.world.sites.site_count),),
        -1,
        dtype=torch.int64,
        device="cpu",
    ).contiguous()
    seed = {
        "positions0_f32_cpu": seed_provider.world.sites.positions0.to(
            device="cpu", dtype=torch.float32
        ).clone().contiguous(),
        "colors_f32_cpu": material_seed.site_rgba_f32[:, :3]
        .clone()
        .contiguous(),
        "source_frame_indices_i64_cpu": source_frame_indices,
        "source_time_provenance": (
            "unavailable_in_xyz_rgb_asset_use_sequence_midpoint_broad_support_v1"
        ),
        "initializer_generation_digest": world_initializer.generation_digest,
        "material_seed_generation_digest": (
            material_seed.material_seed_generation_digest
        ),
        "sites_content_digest": seed_provider.world.sites_content_digest,
        "material_content_digest": material_seed.material_content_digest,
    }
    seed["representation_seed_content_sha256"] = (
        _representation_seed_content_sha256(seed)
    )

    def seal_inputs(dataset: Any) -> Any:
        return seal_worldfoam_public_training_inputs(
            dataset=dataset,
            sample_id=context.protocol.dataset.sample_id,
            dataset_generation_digest=dataset_generation_digest,
            heldout_dataset_generation_digest=heldout_dataset_generation_digest,
            dataset_capability_sha256=capability["capability_sha256"],
            initialization_sha256=context.scene_receipt["initialization_sha256"],
            compiler_sha256=context.scene_receipt["compiler_sha256"],
            same_representation_group=context.route_spec[
                "same_representation_group"
            ],
            target_provider=train_target_provider,
            ray_provider=train_ray_provider,
            heldout_target_provider=heldout_target_provider,
            heldout_ray_provider=heldout_ray_provider,
            frame_times=frame_times,
            world_initializer=world_initializer,
            program_factory=program_factory,
            background_rgb_f32_cpu=torch.tensor(
                runtime["background_rgb"], dtype=torch.float32, device="cpu"
            ).contiguous(),
            artifact_store_policy=artifact_store_policy,
            lazy_memory_policy=lazy_memory_policy,
            full_geometry_memory_policy=full_geometry_memory_policy,
            combined_sgd_policy=combined_sgd_policy,
            material_parameterization=material_parameterization,
            material_sgd_policy=material_sgd_policy,
            maximum_material_state_logical_tensor_bytes=int(
                runtime["maximum_material_state_logical_tensor_bytes"]
            ),
            maximum_tracks_per_bundle=int(runtime["maximum_tracks_per_bundle"]),
            maximum_observations_per_bundle=int(
                runtime["maximum_observations_per_bundle"]
            ),
            maximum_rows_per_native_block=int(
                runtime["maximum_rows_per_native_block"]
            ),
            maximum_samples_per_launch=int(runtime["maximum_samples_per_launch"]),
            maximum_artifact_accounted_bytes_per_entry=int(
                runtime["maximum_artifact_accounted_bytes_per_entry"]
            ),
            cone_tolerance=float(runtime["cone_tolerance"]),
            shared_reverse_mode=str(runtime["shared_reverse_mode"]),
        )

    maximum_source_decode_tensor_bytes = int(
        runtime["lazy_memory_policy"]["max_decoded_frame_scratch_tensor_bytes"]
    )
    required_selected_read_bytes = 70 * max(
        int(context.work_plan.maximum_pixels_per_chunk),
        int(
            getattr(
                context.work_plan,
                "heldout_maximum_pixels_per_chunk",
                context.work_plan.maximum_pixels_per_chunk,
            )
        ),
    )
    if maximum_source_decode_tensor_bytes < required_selected_read_bytes:
        raise MemoryError(
            "frozen selected-pixel source budget cannot admit one row-worker chunk"
        )
    return WorldFoamPublicQualityDataset(
        context=context,
        capability=capability,
        train_target_provider=train_target_provider,
        train_ray_provider=train_ray_provider,
        heldout_target_provider=heldout_target_provider,
        heldout_ray_provider=heldout_ray_provider,
        frame_times=frame_times,
        sealed_inputs_factory=seal_inputs,
        seed=seed,
        maximum_source_decode_tensor_bytes=maximum_source_decode_tensor_bytes,
    )


__all__ = (
    "WorldFoamPublicQualityDataset",
    "create_public_quality_dataset",
)
