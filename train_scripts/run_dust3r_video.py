import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DUST3R_ROOT = ROOT / "third_party" / "dust3r"
if str(DUST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(DUST3R_ROOT))

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.demo import get_3D_model_from_scene
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images


def parse_args():
    parser = argparse.ArgumentParser(description="Run DUSt3R on a sampled video and save outputs.")
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "test_data" / "test_video_small.mp4",
        help="Path to the input video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "test_data" / "dust3r_outputs" / "test_video_small",
        help="Directory where frames and DUSt3R outputs will be saved.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="naver/DUSt3R_ViTLarge_BaseDecoder_224_linear",
        help="DUSt3R checkpoint name or local checkpoint path.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        choices=(224, 512),
        help="DUSt3R input image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "mps", "cpu", "cuda"),
        help="Torch device to use for inference.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=15,
        help="Sample every Nth source frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=24,
        help="Maximum number of sampled frames to process. Use 0 for all sampled frames.",
    )
    parser.add_argument(
        "--scene-graph",
        type=str,
        default="swin",
        choices=("complete", "swin", "logwin", "oneref"),
        help="DUSt3R image-pair graph.",
    )
    parser.add_argument(
        "--winsize",
        type=int,
        default=5,
        help="Window size used by sliding-window scene graphs.",
    )
    parser.add_argument(
        "--noncyclic",
        action="store_true",
        help="Disable cyclic wrap-around for sliding-window scene graphs.",
    )
    parser.add_argument(
        "--refid",
        type=int,
        default=0,
        help="Reference image index for the oneref graph.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=("cosine", "linear"),
        help="Global alignment schedule.",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=100,
        help="Global alignment iterations for multi-frame runs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Global alignment learning rate.",
    )
    parser.add_argument(
        "--min-conf-thr",
        type=float,
        default=3.0,
        help="Confidence threshold used when exporting the GLB scene.",
    )
    return parser.parse_args()


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_scene_graph(scene_graph: str, winsize: int, refid: int, noncyclic: bool) -> str:
    if scene_graph in {"complete"}:
        return scene_graph
    if scene_graph in {"swin", "logwin"}:
        suffix = "-noncyclic" if noncyclic else ""
        return f"{scene_graph}-{winsize}{suffix}"
    if scene_graph == "oneref":
        return f"oneref-{refid}"
    raise ValueError(f"Unsupported scene_graph={scene_graph}")


def extract_sampled_frames(video_path: Path, frames_dir: Path, frame_stride: int, max_frames: int):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("*.png"):
        old_frame.unlink()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sampled = []
    source_frame = 0
    saved_count = 0

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        if source_frame % frame_stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_path = frames_dir / f"frame_{saved_count:04d}_src_{source_frame:05d}.png"
            Image.fromarray(frame_rgb).save(frame_path)
            sampled.append(
                {
                    "sample_idx": saved_count,
                    "source_frame_idx": source_frame,
                    "timestamp_seconds": source_frame / fps if fps > 0 else None,
                    "path": str(frame_path.resolve()),
                }
            )
            saved_count += 1
            if max_frames > 0 and saved_count >= max_frames:
                break
        source_frame += 1

    capture.release()
    if len(sampled) < 2:
        raise ValueError("Need at least 2 sampled frames for DUSt3R.")

    return {
        "fps": fps,
        "total_frames": total_frames,
        "sampled_frames": sampled,
    }


def stack_tensors(values):
    return np.stack([value.detach().cpu().numpy() for value in values], axis=0)


def save_preview_triplets(preview_dir: Path, rgb_images: np.ndarray, depthmaps: np.ndarray, confidences: np.ndarray):
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old_preview in preview_dir.glob("*.png"):
        old_preview.unlink()
    depth_min = float(np.nanmin(depthmaps))
    depth_max = float(np.nanmax(depthmaps))
    conf_min = float(np.nanmin(confidences))
    conf_max = float(np.nanmax(confidences))

    for index in range(rgb_images.shape[0]):
        rgb_image = np.clip(rgb_images[index], 0.0, 1.0)
        Image.fromarray((rgb_image * 255).astype(np.uint8)).save(preview_dir / f"rgb_{index:04d}.png")

        depth = depthmaps[index]
        depth_norm = (depth - depth_min) / max(depth_max - depth_min, 1e-8)
        depth_vis = (cm.viridis(depth_norm)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save(preview_dir / f"depth_{index:04d}.png")

        conf = confidences[index]
        conf_norm = (conf - conf_min) / max(conf_max - conf_min, 1e-8)
        conf_vis = (cm.jet(conf_norm)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(conf_vis).save(preview_dir / f"conf_{index:04d}.png")


def build_per_frame_camera_records(frame_info, poses, intrinsics, focals):
    records = []
    for index, frame in enumerate(frame_info["sampled_frames"]):
        principal_point = intrinsics[index, :2, 2]
        records.append(
            {
                "sample_idx": frame["sample_idx"],
                "source_frame_idx": frame["source_frame_idx"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "frame_path": frame["path"],
                "camera_to_world": poses[index].tolist(),
                "intrinsics": intrinsics[index].tolist(),
                "focal_length_pixels": float(focals[index].reshape(-1)[0]),
                "principal_point_pixels": principal_point.tolist(),
            }
        )
    return records


def make_poses_relative_to_first(poses):
    first_inv = np.linalg.inv(poses[0])
    return np.stack([first_inv @ pose for pose in poses], axis=0)


def main():
    args = parse_args()
    device = pick_device(args.device)

    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using DUSt3R from: {DUST3R_ROOT}")
    print(f"Using device: {device}")
    print(f"Source video: {args.video}")

    frame_info = extract_sampled_frames(args.video, frames_dir, args.frame_stride, args.max_frames)
    frame_paths = [record["path"] for record in frame_info["sampled_frames"]]
    print(f"Extracted {len(frame_paths)} sampled frames to {frames_dir}")

    model = AsymmetricCroCo3DStereo.from_pretrained(args.model_name).to(device)
    square_ok = getattr(model, "square_ok", False)
    images = load_images(
        frame_paths,
        size=args.image_size,
        verbose=True,
        patch_size=model.patch_size,
        square_ok=square_ok,
    )

    scene_graph_spec = build_scene_graph(args.scene_graph, args.winsize, args.refid, args.noncyclic)
    pairs = make_pairs(images, scene_graph=scene_graph_spec, prefilter=None, symmetrize=True)
    print(f"Built {len(pairs)} image pairs with scene graph {scene_graph_spec}")

    output = inference(pairs, model, device, batch_size=1, verbose=True)

    if len(images) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer
    else:
        mode = GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=True)

    alignment_loss = None
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        alignment_loss = float(
            scene.compute_global_alignment(
                init="mst",
                niter=args.niter,
                schedule=args.schedule,
                lr=args.lr,
            )
        )

    scene_glb = get_3D_model_from_scene(
        str(output_dir),
        silent=False,
        scene=scene,
        min_conf_thr=args.min_conf_thr,
        as_pointcloud=True,
        mask_sky=False,
        clean_depth=True,
        transparent_cams=False,
        cam_size=0.05,
    )

    rgb_images = np.stack(scene.imgs, axis=0)
    depthmaps = stack_tensors(scene.get_depthmaps())
    pts3d = stack_tensors(scene.get_pts3d())
    focals = scene.get_focals().detach().cpu().numpy()
    poses = scene.get_im_poses().detach().cpu().numpy()
    intrinsics = scene.get_intrinsics().detach().cpu().numpy()
    masks = np.stack([mask.detach().cpu().numpy() for mask in scene.get_masks()], axis=0)
    confidences = np.stack([conf.detach().cpu().numpy() for conf in scene.im_conf], axis=0)

    np.save(output_dir / "rgb_images.npy", rgb_images)
    np.save(output_dir / "depthmaps.npy", depthmaps)
    np.save(output_dir / "pts3d.npy", pts3d)
    np.save(output_dir / "focals.npy", focals)
    np.save(output_dir / "poses_c2w.npy", poses)
    poses_relative_to_first = make_poses_relative_to_first(poses)
    np.save(output_dir / "poses_c2w_relative_to_first.npy", poses_relative_to_first)
    np.save(output_dir / "intrinsics.npy", intrinsics)
    np.save(output_dir / "confidence_masks.npy", masks)
    np.save(output_dir / "confidences.npy", confidences)
    np.savez(
        output_dir / "camera_bundle.npz",
        poses_c2w=poses,
        poses_c2w_relative_to_first=poses_relative_to_first,
        intrinsics=intrinsics,
        focals=focals,
    )

    save_preview_triplets(preview_dir, rgb_images, depthmaps, confidences)
    per_frame_cameras = build_per_frame_camera_records(frame_info, poses, intrinsics, focals)
    (output_dir / "per_frame_cameras.json").write_text(json.dumps(per_frame_cameras, indent=2))

    summary = {
        "video": str(args.video.resolve()),
        "device": device,
        "model_name": args.model_name,
        "image_size": args.image_size,
        "scene_graph": scene_graph_spec,
        "pair_count": len(pairs),
        "mode": mode.value,
        "alignment_loss": alignment_loss,
        "scene_glb": str(Path(scene_glb).resolve()) if scene_glb is not None else None,
        "frame_sampling": frame_info,
        "per_frame_cameras_path": str((output_dir / "per_frame_cameras.json").resolve()),
        "camera_bundle_path": str((output_dir / "camera_bundle.npz").resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved DUSt3R outputs to {output_dir}")
    print(f"Scene GLB: {scene_glb}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
