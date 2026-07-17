from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image

try:
    from .report_artifacts import ensure_train_path, validate_frame_indices, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ensure_train_path, validate_frame_indices, write_report_json

ensure_train_path()

from config_utils import apply_defaults, load_config_file, resolved_config
from multicam_video_data import load_multicam_video_bundle

pycolmap: Any | None = None


def require_pycolmap() -> Any:
    global pycolmap
    if pycolmap is None:
        try:
            import pycolmap as imported_pycolmap  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "build_pycolmap_known_pose_point_cloud.py requires pycolmap for execution. "
                "Install pycolmap or run this script in the Modal/Colmap environment."
            ) from exc
        pycolmap = imported_pycolmap
    return pycolmap


GEOMETRY_DATA_DEFAULTS = {
    "video_path": None,
    "frame_source": "multicam_val",
    "max_frames": 16,
}


def resolve_geometry_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = resolved_config(config, ("data", "model"))
    cfg.setdefault("camera", {})
    apply_defaults(cfg["data"], GEOMETRY_DATA_DEFAULTS)
    if cfg["data"]["frame_source"] != "multicam_val":
        raise ValueError("Known-pose pycolmap point cloud builder expects data.frame_source='multicam_val'.")
    for key in ("xy_extent", "z_min", "z_max"):
        if key not in cfg["model"]:
            raise KeyError(f"Missing required model.{key} in geometry config.")
    return cfg


def write_rgb_image(path: Path, image: torch.Tensor) -> None:
    rgb = (image.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(rgb).save(path)


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors_u8 = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as fh:
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write(f"element vertex {int(points.shape[0])}\n")
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write("property uchar red\n")
        fh.write("property uchar green\n")
        fh.write("property uchar blue\n")
        fh.write("end_header\n")
        for point, color in zip(points.tolist(), colors_u8.tolist()):
            fh.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def int_histogram(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {}
    unique, counts = np.unique(values.astype(np.int64), return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(unique.tolist(), counts.tolist())}


def apply_option_overrides(target: Any, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if value is not None:
            setattr(target, key, value)


def parse_pycolmap_enum(enum_cls: Any, raw_value: str) -> Any:
    name = raw_value.upper()
    if not hasattr(enum_cls, name):
        valid = sorted(key.lower() for key in dir(enum_cls) if key.isupper())
        raise ValueError(f"Unknown {enum_cls.__name__} {raw_value!r}; expected one of {valid}.")
    return getattr(enum_cls, name)


def resolve_pycolmap_device(raw_device: str) -> Any:
    name = raw_device.lower()
    if not hasattr(pycolmap.Device, name):
        valid = sorted(key for key in ("auto", "cpu", "cuda") if hasattr(pycolmap.Device, key))
        raise ValueError(f"Unknown pycolmap device {raw_device!r}; expected one of {valid}.")
    return getattr(pycolmap.Device, name)


def parse_frame_indices(args: argparse.Namespace, *, frame_count: int) -> list[int]:
    if args.frame_indices is None:
        indices = [int(args.frame_index)]
    else:
        indices = [int(index) for index in args.frame_indices]
    return validate_frame_indices(indices, frame_count=frame_count)


def image_name_for(camera_name: str, frame_index: int, *, multi_frame: bool) -> str:
    if not multi_frame:
        return f"{camera_name}.png"
    return f"{camera_name}_frame{int(frame_index):04d}.png"


def camera_params_for_view(bundle: Any, view_index: int, camera_model: str) -> np.ndarray:
    K = bundle.train_K[view_index].detach().cpu().numpy().astype(np.float64)
    params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    if camera_model == "PINHOLE":
        return np.array(params, dtype=np.float64)
    if camera_model == "OPENCV_FISHEYE":
        if bundle.train_distortions is None:
            raise ValueError("OPENCV_FISHEYE camera model requires bundle.train_distortions.")
        distortion = bundle.train_distortions[view_index].detach().cpu().numpy().astype(np.float64).reshape(-1)
        if distortion.size < 4:
            distortion = np.pad(distortion, (0, 4 - distortion.size))
        return np.array([*params, *distortion[:4]], dtype=np.float64)
    raise ValueError(f"Unsupported pycolmap camera model {camera_model!r}.")


def resolve_pycolmap_camera_model(raw_model: str, bundle: Any) -> str:
    model = raw_model.lower()
    if model == "auto":
        lens_models = set(bundle.train_lens_models or [])
        model = "opencv_fisheye" if lens_models == {"opencv_fisheye"} else "pinhole"
    if model == "pinhole":
        return "PINHOLE"
    if model == "opencv_fisheye":
        return "OPENCV_FISHEYE"
    raise ValueError("camera_model must be 'auto', 'pinhole', or 'opencv_fisheye'")


def camera_params_csv(params: np.ndarray) -> str:
    return ",".join(str(float(value)) for value in params)


def build_reconstruction_from_bundle(
    *,
    database: pycolmap.Database,
    bundle: Any,
    camera_model: str,
    target_size: int,
    image_records: dict[str, tuple[int, int]],
) -> pycolmap.Reconstruction:
    reconstruction = pycolmap.Reconstruction()
    added_cameras: dict[int, tuple[int, np.ndarray]] = {}
    for image_db in database.read_all_images():
        if image_db.name not in image_records:
            raise KeyError(f"Database image {image_db.name!r} does not match any generated train image.")
        view_index, frame_index = image_records[image_db.name]
        camera_id = int(image_db.camera_id)
        camera_params = camera_params_for_view(bundle, view_index, camera_model)
        if camera_id in added_cameras:
            existing_view, existing_params = added_cameras[camera_id]
            if not np.allclose(existing_params, camera_params, rtol=1.0e-5, atol=1.0e-5):
                raise ValueError(
                    "pycolmap database shared camera_id "
                    f"{camera_id} between train views {existing_view} and {view_index} with different intrinsics. "
                    "Use --camera-mode per_image or train cameras with matching K."
                )
        else:
            camera = pycolmap.Camera(
                camera_id=camera_id,
                model=camera_model,
                width=int(target_size),
                height=int(target_size),
                params=camera_params,
            )
            reconstruction.add_camera_with_trivial_rig(camera)
            added_cameras[camera_id] = (view_index, camera_params)
        image = pycolmap.Image(
            name=str(image_db.name),
            camera_id=camera_id,
            image_id=int(image_db.image_id),
        )
        w2c = bundle.train_w2c[view_index, frame_index].detach().cpu().numpy().astype(np.float64)
        reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d(w2c[:3]))
    return reconstruction


def build_reconstruction_from_image_records(
    *,
    bundle: Any,
    camera_model: str,
    camera_mode: pycolmap.CameraMode,
    target_size: int,
    image_records: dict[str, tuple[int, int]],
) -> pycolmap.Reconstruction:
    reconstruction = pycolmap.Reconstruction()
    camera_ids_by_view: dict[int, int] = {}
    shared_camera_id = 1
    if camera_mode == pycolmap.CameraMode.SINGLE:
        view_indices = sorted({int(view_index) for view_index, _frame_index in image_records.values()})
        reference = camera_params_for_view(bundle, view_indices[0], camera_model)
        for view_index in view_indices[1:]:
            params = camera_params_for_view(bundle, view_index, camera_model)
            if not np.allclose(reference, params, rtol=1.0e-5, atol=1.0e-5):
                raise ValueError(
                    "CameraMode.SINGLE requested for train views with different intrinsics. "
                    "Use --camera-mode per_image or train cameras with matching K."
                )
    for image_id, (image_name, (view_index, frame_index)) in enumerate(image_records.items(), start=1):
        if camera_mode == pycolmap.CameraMode.SINGLE:
            camera_id = shared_camera_id
        else:
            camera_id = camera_ids_by_view.setdefault(int(view_index), int(view_index) + 1)
        if camera_id not in reconstruction.cameras:
            camera = pycolmap.Camera(
                camera_id=camera_id,
                model=camera_model,
                width=int(target_size),
                height=int(target_size),
                params=camera_params_for_view(bundle, int(view_index), camera_model),
            )
            reconstruction.add_camera_with_trivial_rig(camera)
        image = pycolmap.Image(name=str(image_name), camera_id=camera_id, image_id=int(image_id))
        w2c = bundle.train_w2c[view_index, frame_index].detach().cpu().numpy().astype(np.float64)
        reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d(w2c[:3]))
    return reconstruction


def resolve_camera_mode(raw_mode: str, camera_params_by_view: list[np.ndarray]) -> pycolmap.CameraMode:
    mode = raw_mode.lower()
    if mode == "auto":
        mode = (
            "single"
            if all(np.allclose(params, camera_params_by_view[0], rtol=1.0e-5, atol=1.0e-5) for params in camera_params_by_view)
            else "per_image"
        )
    if mode == "single":
        return pycolmap.CameraMode.SINGLE
    if mode == "per_image":
        return pycolmap.CameraMode.PER_IMAGE
    raise ValueError("camera_mode must be 'auto', 'single', or 'per_image'")


def make_two_view_geometry_options(args: argparse.Namespace) -> pycolmap.TwoViewGeometryOptions:
    options = pycolmap.TwoViewGeometryOptions()
    if args.verify_max_error is not None:
        options.ransac.max_error = float(args.verify_max_error)
        options.watermark_detection_max_error = float(args.verify_max_error)
        options.stationary_matches_max_error = float(args.verify_max_error)
    apply_option_overrides(
        options,
        {
            "detect_watermark": args.verify_detect_watermark,
            "filter_stationary_matches": args.verify_filter_stationary_matches,
        },
    )
    return options


def require_onnx_opt_in(args: argparse.Namespace) -> None:
    if str(args.feature_backend) == "hloc":
        return
    feature_type = str(args.feature_type).lower()
    matcher_type = str(args.matcher_type).lower()
    needs_onnx = "aliked" in feature_type or "aliked" in matcher_type or "lightglue" in matcher_type
    if needs_onnx and not bool(args.allow_onnx_models):
        raise RuntimeError(
            "ALIKED/LightGlue pycolmap modes require an ONNX-enabled pycolmap build. "
            "The local Mac pycolmap wheel aborts inside C++ without ONNX support; "
            "rerun on an ONNX-enabled host with --allow-onnx-models."
        )


def hloc_feature_conf(feature_type: str) -> dict[str, Any]:
    from hloc import extract_features  # type: ignore

    feature = feature_type.lower()
    model_names = {
        "aliked_n16rot": "aliked-n16rot",
        "aliked_n32": "aliked-n32",
    }
    if feature not in model_names:
        raise ValueError("feature_backend='hloc' supports --feature-type aliked_n16rot or aliked_n32.")
    conf = copy.deepcopy(extract_features.confs["aliked-n16"])
    model_name = model_names[feature]
    conf.setdefault("model", {})["model_name"] = model_name
    conf["output"] = f"feats-{model_name}"
    return conf


def hloc_match_conf(matcher_type: str) -> dict[str, Any]:
    from hloc import match_features  # type: ignore

    matcher = matcher_type.lower()
    if matcher == "aliked_lightglue":
        return copy.deepcopy(match_features.confs["aliked+lightglue"])
    raise ValueError("feature_backend='hloc' currently supports --matcher-type aliked_lightglue.")


def feature_type_for_colmap_cli(feature_type: str) -> str:
    return feature_type.upper()


def matcher_type_for_colmap_cli(matcher_type: str) -> str:
    return matcher_type.upper()


def write_exhaustive_pairs(path: Path, image_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, name0 in enumerate(image_names):
            for name1 in image_names[i + 1 :]:
                fh.write(f"{name0} {name1}\n")


def run_hloc_import_backend(
    *,
    database_path: Path,
    images_dir: Path,
    workdir: Path,
    bundle: Any,
    camera_model: str,
    camera_mode: pycolmap.CameraMode,
    target_size: int,
    image_names: list[str],
    image_records: dict[str, tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[pycolmap.Reconstruction, dict[str, Any]]:
    try:
        from hloc import extract_features, match_features, triangulation  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "feature_backend='hloc' requires Hierarchical-Localization. "
            "Run with: KMP_DUPLICATE_LIB_OK=TRUE uv run --with "
            "'git+https://github.com/cvg/Hierarchical-Localization.git' ..."
        ) from exc

    reconstruction = build_reconstruction_from_image_records(
        bundle=bundle,
        camera_model=camera_model,
        camera_mode=camera_mode,
        target_size=target_size,
        image_records=image_records,
    )
    pairs_path = workdir / "pairs-exhaustive.txt"
    write_exhaustive_pairs(pairs_path, image_names)

    feature_conf = hloc_feature_conf(str(args.feature_type))
    feature_conf["preprocessing"] = dict(feature_conf.get("preprocessing", {}))
    feature_conf["preprocessing"]["resize_max"] = int(target_size)
    feature_conf.setdefault("model", {})["max_num_keypoints"] = (
        int(args.max_features) if int(args.max_features) > 0 else -1
    )
    match_conf = hloc_match_conf(str(args.matcher_type))
    features_path = extract_features.main(
        feature_conf,
        images_dir,
        export_dir=workdir,
        image_list=image_names,
        overwrite=True,
    )
    matches_path = match_features.main(
        match_conf,
        pairs_path,
        str(feature_conf["output"]),
        export_dir=workdir,
        overwrite=True,
    )
    image_ids = triangulation.create_db_from_model(reconstruction, database_path)
    database = pycolmap.Database.open(database_path)
    try:
        triangulation.import_features(image_ids, database, features_path)
        verification_max_error = float(args.verify_max_error or args.max_reproj_error)
        triangulation.import_matches(image_ids, database, pairs_path, matches_path)
        triangulation.geometric_verification(
            image_ids,
            reconstruction,
            database,
            features_path,
            pairs_path,
            matches_path,
            max_error=verification_max_error,
        )
    finally:
        database.close()

    return reconstruction, {
        "hloc_pairs_path": str(pairs_path),
        "hloc_features_path": str(features_path),
        "hloc_matches_path": str(matches_path),
        "hloc_feature_conf": feature_conf,
        "hloc_match_conf": match_conf,
        "hloc_forced_known_pose_guided_verification": True,
        "hloc_feature_model_name": feature_conf["model"].get("model_name"),
        "hloc_feature_max_num_keypoints": feature_conf["model"].get("max_num_keypoints"),
        "known_pose_verification_applied": True,
        "known_pose_verification_backend": "hloc.geometric_verification",
        "known_pose_verification_max_error": verification_max_error,
    }


def run_logged_command(command: list[str], *, cwd: Path, timeout_s: int | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    result = {
        "command": command,
        "return_code": int(completed.returncode),
        "stdout_tail": completed.stdout[-12000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_colmap_cli_backend(
    *,
    database_path: Path,
    images_dir: Path,
    bundle: Any,
    camera_model: str,
    camera_mode: pycolmap.CameraMode,
    target_size: int,
    image_records: dict[str, tuple[int, int]],
    camera_params_by_view: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[pycolmap.Reconstruction, dict[str, Any]]:
    feature_type = feature_type_for_colmap_cli(str(args.feature_type))
    matcher_type = matcher_type_for_colmap_cli(str(args.matcher_type))
    use_gpu = "1" if bool(args.pycolmap_use_gpu) else "0"
    single_camera = "1" if camera_mode == pycolmap.CameraMode.SINGLE else "0"
    feature_command = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.camera_params",
        camera_params_csv(camera_params_by_view[0]),
        "--ImageReader.single_camera",
        single_camera,
        "--FeatureExtraction.type",
        feature_type,
        "--FeatureExtraction.use_gpu",
        use_gpu,
    ]
    if feature_type.startswith("ALIKED"):
        feature_command.extend(["--AlikedExtraction.max_num_features", str(int(args.max_features))])
    else:
        feature_command.extend(["--SiftExtraction.max_num_features", str(int(args.max_features))])

    match_command = [
        "colmap",
        "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--FeatureMatching.type",
        matcher_type,
        "--FeatureMatching.use_gpu",
        use_gpu,
    ]
    if matcher_type.startswith("SIFT"):
        match_command.extend(["--SiftMatching.max_ratio", str(float(args.sift_ratio))])
        if args.sift_max_distance is not None:
            match_command.extend(["--SiftMatching.max_distance", str(float(args.sift_max_distance))])
        if args.sift_cross_check is not None:
            match_command.extend(["--SiftMatching.cross_check", "1" if bool(args.sift_cross_check) else "0"])

    feature_result = run_logged_command(feature_command, cwd=ROOT, timeout_s=60 * 60)
    match_result = run_logged_command(match_command, cwd=ROOT, timeout_s=60 * 60)
    database = pycolmap.Database.open(database_path)
    try:
        reconstruction = build_reconstruction_from_bundle(
            database=database,
            bundle=bundle,
            camera_model=camera_model,
            target_size=target_size,
            image_records=image_records,
        )
    finally:
        database.close()
    verification_applied = bool(args.known_pose_guided_verification)
    verification_max_error = float(args.verify_max_error or args.max_reproj_error)
    if verification_applied:
        pycolmap.guided_geometric_verification(
            reconstruction,
            database_path,
            two_view_geometry_options=make_two_view_geometry_options(args),
        )
    return reconstruction, {
        "colmap_cli_feature_command": feature_result["command"],
        "colmap_cli_feature_stdout_tail": feature_result["stdout_tail"],
        "colmap_cli_match_command": match_result["command"],
        "colmap_cli_match_stdout_tail": match_result["stdout_tail"],
        "known_pose_verification_applied": verification_applied,
        "known_pose_verification_backend": (
            "colmap_cli_exhaustive_matcher+pycolmap.guided_geometric_verification"
            if verification_applied
            else "colmap_cli_exhaustive_matcher"
        ),
        "known_pose_verification_max_error": verification_max_error if verification_applied else None,
    }


def build_pycolmap_cloud(args: argparse.Namespace) -> dict[str, Any]:
    require_pycolmap()
    require_onnx_opt_in(args)
    cfg = resolve_geometry_config(load_config_file(args.config))
    if args.train_cameras:
        cfg["data"]["multicam_train_cameras"] = [str(camera) for camera in args.train_cameras]
    if args.heldout_camera is not None:
        cfg["data"]["multicam_heldout_camera"] = str(args.heldout_camera)
        cfg["data"]["multicam_heldout_cameras"] = None
    if args.anchor_camera is not None:
        cfg["data"]["multicam_anchor_camera"] = str(args.anchor_camera)
    if args.condition_camera is not None:
        cfg["data"]["multicam_condition_camera"] = str(args.condition_camera)
    target_size = int(args.target_size)
    xy_extent = float(args.xy_extent if args.xy_extent is not None else cfg["model"]["xy_extent"])
    z_min = float(args.z_min if args.z_min is not None else cfg["model"]["z_min"])
    z_max = float(args.z_max if args.z_max is not None else cfg["model"]["z_max"])
    workdir = Path(args.workdir) if args.workdir else None
    temp_ctx = None
    if workdir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix="powerfoam_pycolmap_")
        workdir = Path(temp_ctx.name)
    if workdir.exists():
        shutil.rmtree(workdir)
    images_dir = workdir / "images"
    sparse_dir = workdir / "sparse"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    database_path = workdir / "database.db"

    bundle = load_multicam_video_bundle(
        data_cfg=cfg["data"],
        camera_cfg=cfg["camera"],
        target_size=target_size,
        device=torch.device("cpu"),
    )
    frame_indices = parse_frame_indices(args, frame_count=bundle.frame_count)
    multi_frame = len(frame_indices) > 1
    if bundle.train_view_count < 2:
        raise ValueError("Known-pose pycolmap triangulation requires at least two train cameras.")

    image_names = []
    image_records: dict[str, tuple[int, int]] = {}
    for view_index, camera_name in enumerate(bundle.train_camera_names):
        for frame_index in frame_indices:
            image_name = image_name_for(camera_name, frame_index, multi_frame=multi_frame)
            write_rgb_image(
                images_dir / image_name,
                bundle.train_frames[view_index, frame_index].to(dtype=torch.float32),
            )
            image_names.append(image_name)
            image_records[image_name] = (view_index, frame_index)

    camera_model = resolve_pycolmap_camera_model(str(args.camera_model), bundle)
    camera_params_by_view = [
        camera_params_for_view(bundle, view_index, camera_model)
        for view_index in range(bundle.train_view_count)
    ]
    camera_mode = resolve_camera_mode(str(args.camera_mode), camera_params_by_view)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    reader.camera_params = camera_params_csv(camera_params_by_view[0])
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.type = parse_pycolmap_enum(pycolmap.FeatureExtractorType, str(args.feature_type))
    extraction.use_gpu = bool(args.pycolmap_use_gpu)
    extraction.max_image_size = target_size
    extraction.sift.max_num_features = int(args.max_features)
    apply_option_overrides(
        extraction.sift,
        {
            "peak_threshold": args.sift_peak_threshold,
            "edge_threshold": args.sift_edge_threshold,
            "estimate_affine_shape": args.sift_estimate_affine_shape,
            "domain_size_pooling": args.sift_domain_size_pooling,
        },
    )
    matching = pycolmap.FeatureMatchingOptions()
    matching.type = parse_pycolmap_enum(pycolmap.FeatureMatcherType, str(args.matcher_type))
    matching.use_gpu = bool(args.pycolmap_use_gpu)
    matching.guided_matching = bool(args.guided_matching)
    matching.sift.max_ratio = float(args.sift_ratio)
    apply_option_overrides(
        matching.sift,
        {
            "max_distance": args.sift_max_distance,
            "cross_check": args.sift_cross_check,
        },
    )
    verification = make_two_view_geometry_options(args)
    pycolmap_device = resolve_pycolmap_device(str(args.pycolmap_device))

    backend_extra: dict[str, Any] = {}
    if str(args.feature_backend) == "pycolmap":
        pycolmap.extract_features(
            database_path,
            images_dir,
            image_names=image_names,
            camera_mode=camera_mode,
            reader_options=reader,
            extraction_options=extraction,
            device=pycolmap_device,
        )
        pycolmap.match_exhaustive(
            database_path,
            matching_options=matching,
            verification_options=verification,
            device=pycolmap_device,
        )
        database = pycolmap.Database.open(database_path)
        reconstruction = build_reconstruction_from_bundle(
            database=database,
            bundle=bundle,
            camera_model=camera_model,
            target_size=target_size,
            image_records=image_records,
        )
        database.close()
        if bool(args.known_pose_guided_verification):
            pycolmap.guided_geometric_verification(
                reconstruction,
                database_path,
                two_view_geometry_options=verification,
            )
    elif str(args.feature_backend) == "hloc":
        reconstruction, backend_extra = run_hloc_import_backend(
            database_path=database_path,
            images_dir=images_dir,
            workdir=workdir,
            bundle=bundle,
            camera_model=camera_model,
            camera_mode=camera_mode,
            target_size=target_size,
            image_names=image_names,
            image_records=image_records,
            args=args,
        )
    elif str(args.feature_backend) == "colmap_cli":
        reconstruction, backend_extra = run_colmap_cli_backend(
            database_path=database_path,
            images_dir=images_dir,
            bundle=bundle,
            camera_model=camera_model,
            camera_mode=camera_mode,
            target_size=target_size,
            image_records=image_records,
            camera_params_by_view=camera_params_by_view,
            args=args,
        )
    else:
        raise ValueError("feature_backend must be 'pycolmap', 'hloc', or 'colmap_cli'.")
    options = pycolmap.IncrementalPipelineOptions()
    options.mapper.filter_max_reproj_error = float(args.max_reproj_error)
    options.mapper.filter_min_tri_angle = float(args.min_tri_angle)
    options.triangulation.min_angle = float(args.min_tri_angle)
    options.triangulation.ignore_two_view_tracks = bool(args.ignore_two_view_tracks)
    options.triangulation.create_max_angle_error = float(args.max_reproj_error)
    options.triangulation.continue_max_angle_error = float(args.max_reproj_error)
    options.triangulation.complete_max_reproj_error = float(args.max_reproj_error)
    apply_option_overrides(
        options.triangulation,
        {
            "max_transitivity": args.triangulation_max_transitivity,
            "complete_max_transitivity": args.triangulation_complete_max_transitivity,
            "merge_max_reproj_error": args.triangulation_merge_max_reproj_error,
            "complete_max_reproj_error": args.triangulation_complete_max_reproj_error,
            "re_max_angle_error": args.triangulation_re_max_angle_error,
            "re_max_trials": args.triangulation_re_max_trials,
            "re_min_ratio": args.triangulation_re_min_ratio,
        },
    )

    output_reconstruction = pycolmap.triangulate_points(
        reconstruction,
        database_path,
        images_dir,
        sparse_dir,
        clear_points=True,
        options=options,
        refine_intrinsics=False,
    )

    raw_points = []
    raw_colors = []
    raw_errors = []
    raw_track_lengths = []
    raw_unique_camera_lengths = []
    raw_unique_frame_lengths = []
    image_id_records = {
        int(image.image_id): image_records[str(image.name)]
        for image in output_reconstruction.images.values()
        if str(image.name) in image_records
    }
    for point in output_reconstruction.points3D.values():
        track_records = [
            image_id_records[int(element.image_id)]
            for element in point.track.elements
            if int(element.image_id) in image_id_records
        ]
        raw_points.append(np.asarray(point.xyz, dtype=np.float64))
        raw_colors.append(np.asarray(point.color, dtype=np.float64))
        raw_errors.append(float(point.error))
        raw_track_lengths.append(int(len(point.track.elements)))
        raw_unique_camera_lengths.append(len({view_index for view_index, _frame_index in track_records}))
        raw_unique_frame_lengths.append(len({frame_index for _view_index, frame_index in track_records}))
    if raw_points:
        points = np.stack(raw_points, axis=0)
        colors = np.stack(raw_colors, axis=0)
        errors = np.asarray(raw_errors, dtype=np.float64)
        track_lengths = np.asarray(raw_track_lengths, dtype=np.float64)
        unique_camera_lengths = np.asarray(raw_unique_camera_lengths, dtype=np.float64)
        unique_frame_lengths = np.asarray(raw_unique_frame_lengths, dtype=np.float64)
    else:
        points = np.empty((0, 3), dtype=np.float64)
        colors = np.empty((0, 3), dtype=np.float64)
        errors = np.empty((0,), dtype=np.float64)
        track_lengths = np.empty((0,), dtype=np.float64)
        unique_camera_lengths = np.empty((0,), dtype=np.float64)
        unique_frame_lengths = np.empty((0,), dtype=np.float64)

    finite = np.isfinite(points).all(axis=-1)
    box = (
        finite
        & (np.abs(points[:, 0]) <= xy_extent)
        & (np.abs(points[:, 1]) <= xy_extent)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    min_track_length = int(args.min_track_length)
    min_unique_cameras = int(args.min_unique_cameras)
    track_filter = box & (track_lengths >= min_track_length) & (unique_camera_lengths >= min_unique_cameras)
    filtered_points = points[track_filter]
    filtered_colors = colors[track_filter]
    filtered_errors = errors[track_filter]
    filtered_track_lengths = track_lengths[track_filter]
    filtered_unique_camera_lengths = unique_camera_lengths[track_filter]
    filtered_unique_frame_lengths = unique_frame_lengths[track_filter]
    order = np.argsort(filtered_errors, kind="stable")
    if int(args.max_points) > 0:
        order = order[: int(args.max_points)]
    filtered_points = filtered_points[order]
    filtered_colors = filtered_colors[order]
    filtered_errors = filtered_errors[order]
    filtered_track_lengths = filtered_track_lengths[order]
    filtered_unique_camera_lengths = filtered_unique_camera_lengths[order]
    filtered_unique_frame_lengths = filtered_unique_frame_lengths[order]

    write_ascii_ply(Path(args.output), filtered_points, filtered_colors)
    summary_database = pycolmap.Database.open(database_path)
    summary = {
        "config": str(args.config),
        "output": str(args.output),
        "sample_id": str(bundle.metadata.get("sample_id")) if bundle.metadata else None,
        "train_cameras": list(bundle.train_camera_names),
        "heldout_cameras": list(bundle.heldout_camera_names or []),
        "pose_source": bundle.pose_source,
        "coordinate_frame": "model",
        "frame_index": frame_indices[0] if len(frame_indices) == 1 else None,
        "frame_indices": frame_indices,
        "multi_frame_database": multi_frame,
        "target_size": target_size,
        "camera_model": camera_model,
        "camera_mode": str(args.camera_mode),
        "resolved_camera_mode": camera_mode.name.lower(),
        "camera_params": reader.camera_params,
        "camera_params_by_view": {
            str(camera_name): [float(value) for value in camera_params_by_view[index]]
            for index, camera_name in enumerate(bundle.train_camera_names)
        },
        "max_features": int(args.max_features),
        "feature_backend": str(args.feature_backend),
        "feature_type": str(extraction.type.name).lower(),
        "matcher_type": str(matching.type.name).lower(),
        "allow_onnx_models": bool(args.allow_onnx_models),
        "pycolmap_use_gpu": bool(args.pycolmap_use_gpu),
        "pycolmap_device": str(args.pycolmap_device),
        "sift_ratio": float(args.sift_ratio),
        "guided_matching": bool(args.guided_matching),
        "known_pose_guided_verification": bool(args.known_pose_guided_verification) or str(args.feature_backend) == "hloc",
        "requested_known_pose_guided_verification": bool(args.known_pose_guided_verification),
        "verification_options": {
            "ransac_max_error": float(verification.ransac.max_error),
            "detect_watermark": bool(verification.detect_watermark),
            "filter_stationary_matches": bool(verification.filter_stationary_matches),
        },
        "max_reproj_error": float(args.max_reproj_error),
        "min_tri_angle": float(args.min_tri_angle),
        "ignore_two_view_tracks": bool(args.ignore_two_view_tracks),
        "min_track_length": min_track_length,
        "min_unique_cameras": min_unique_cameras,
        "sift_options": {
            "peak_threshold": float(extraction.sift.peak_threshold),
            "edge_threshold": float(extraction.sift.edge_threshold),
            "estimate_affine_shape": bool(extraction.sift.estimate_affine_shape),
            "domain_size_pooling": bool(extraction.sift.domain_size_pooling),
            "max_distance": float(matching.sift.max_distance),
            "cross_check": bool(matching.sift.cross_check),
        },
        "triangulation_options": {
            "max_transitivity": int(options.triangulation.max_transitivity),
            "complete_max_transitivity": int(options.triangulation.complete_max_transitivity),
            "merge_max_reproj_error": float(options.triangulation.merge_max_reproj_error),
            "complete_max_reproj_error": float(options.triangulation.complete_max_reproj_error),
            "re_max_angle_error": float(options.triangulation.re_max_angle_error),
            "re_max_trials": int(options.triangulation.re_max_trials),
            "re_min_ratio": float(options.triangulation.re_min_ratio),
        },
        "database_num_cameras": int(summary_database.num_cameras()),
        "database_num_images": int(summary_database.num_images()),
        "database_num_keypoints": int(summary_database.num_keypoints()),
        "database_num_matched_image_pairs": int(summary_database.num_matched_image_pairs()),
        "database_num_verified_image_pairs": int(summary_database.num_verified_image_pairs()),
        "raw_point_count": int(points.shape[0]),
        "box_filtered_count": int(box.sum()),
        "track_filtered_count": int(track_filter.sum()),
        "point_count": int(filtered_points.shape[0]),
        "raw_reproj_error": stats(errors),
        "filtered_reproj_error": stats(filtered_errors),
        "raw_track_length": stats(track_lengths),
        "filtered_track_length": stats(filtered_track_lengths),
        "raw_track_length_histogram": int_histogram(track_lengths),
        "filtered_track_length_histogram": int_histogram(filtered_track_lengths),
        "raw_unique_camera_track_length": stats(unique_camera_lengths),
        "filtered_unique_camera_track_length": stats(filtered_unique_camera_lengths),
        "raw_unique_camera_track_length_histogram": int_histogram(unique_camera_lengths),
        "filtered_unique_camera_track_length_histogram": int_histogram(filtered_unique_camera_lengths),
        "raw_unique_frame_track_length": stats(unique_frame_lengths),
        "filtered_unique_frame_track_length": stats(filtered_unique_frame_lengths),
        "raw_unique_frame_track_length_histogram": int_histogram(unique_frame_lengths),
        "filtered_unique_frame_track_length_histogram": int_histogram(filtered_unique_frame_lengths),
        "xy_extent": xy_extent,
        "z_min": z_min,
        "z_max": z_max,
        "workdir": str(workdir) if args.keep_workdir or args.workdir else None,
        **backend_extra,
    }
    summary_database.close()
    write_report_json(Path(args.output).with_suffix(".json"), summary)
    if temp_ctx is not None:
        temp_ctx.cleanup()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a known-pose pycolmap/COLMAP point-cloud init from train cameras.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--frame-indices", nargs="+", type=int, default=None)
    parser.add_argument("--train-cameras", nargs="+", default=None)
    parser.add_argument("--heldout-camera", default=None)
    parser.add_argument("--anchor-camera", default=None)
    parser.add_argument("--condition-camera", default=None)
    parser.add_argument("--camera-model", choices=["auto", "pinhole", "opencv_fisheye"], default="auto")
    parser.add_argument("--camera-mode", choices=["auto", "single", "per_image"], default="auto")
    parser.add_argument("--feature-backend", choices=["pycolmap", "hloc", "colmap_cli"], default="pycolmap")
    parser.add_argument(
        "--feature-type",
        choices=["sift", "aliked_n16rot", "aliked_n32"],
        default="sift",
    )
    parser.add_argument(
        "--matcher-type",
        choices=["sift_bruteforce", "sift_lightglue", "aliked_bruteforce", "aliked_lightglue"],
        default="sift_bruteforce",
    )
    parser.add_argument("--allow-onnx-models", action="store_true")
    parser.add_argument("--pycolmap-use-gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pycolmap-device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--sift-ratio", type=float, default=0.9)
    parser.add_argument("--sift-peak-threshold", type=float, default=None)
    parser.add_argument("--sift-edge-threshold", type=float, default=None)
    parser.add_argument("--sift-estimate-affine-shape", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sift-domain-size-pooling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sift-max-distance", type=float, default=None)
    parser.add_argument("--sift-cross-check", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--guided-matching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-max-error", type=float, default=None)
    parser.add_argument("--verify-detect-watermark", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verify-filter-stationary-matches", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--known-pose-guided-verification", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-reproj-error", type=float, default=8.0)
    parser.add_argument("--min-tri-angle", type=float, default=0.1)
    parser.add_argument("--ignore-two-view-tracks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-track-length", type=int, default=1)
    parser.add_argument("--min-unique-cameras", type=int, default=1)
    parser.add_argument("--triangulation-max-transitivity", type=int, default=None)
    parser.add_argument("--triangulation-complete-max-transitivity", type=int, default=None)
    parser.add_argument("--triangulation-merge-max-reproj-error", type=float, default=None)
    parser.add_argument("--triangulation-complete-max-reproj-error", type=float, default=None)
    parser.add_argument("--triangulation-re-max-angle-error", type=float, default=None)
    parser.add_argument("--triangulation-re-max-trials", type=int, default=None)
    parser.add_argument("--triangulation-re-min-ratio", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--xy-extent", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    summary = build_pycolmap_cloud(parser.parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
