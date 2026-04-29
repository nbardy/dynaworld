"""Runs INSIDE Blender. Opens a .blend, renders N frames, exports c2w + K JSON.

Invoked as:
    blender -b <path>.blend -P render_scene.py -- \
        --out-dir /path/to/out --num-frames 8 --resolution 256 [--camera <name>]

For v0 we use the .blend's existing active camera (the production cinematographer's
shot). Programmatic trajectory variation lands in `trajectories.py` next.

Outputs into <out-dir>:
    frame_0000.png ... frame_NNNN.png    -- rendered RGB
    cameras.json                         -- per-frame K + c2w in OpenCV convention
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import bpy  # type: ignore[import-not-found]  # provided by Blender runtime
import numpy as np

# Import our standalone camera-conversion module. Blender runs this script in
# its own Python context, so we add the package dir to sys.path before import.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
from camera_export import (  # noqa: E402
    blender_world_matrix_to_opencv_c2w,
    intrinsics_from_blender_camera,
)


def parse_args(argv: list[str]) -> dict:
    """Parse args appearing after `--` on the Blender CLI.

    Avoids argparse to keep the runtime small inside Blender.
    """
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    out: dict = {
        "out_dir": None,
        "num_frames": None,         # if None, render scene's full frame range
        "frame_start": None,        # if None, use scene.frame_start
        "downscale": 8,             # divide scene resolution by this
        "camera": None,
        "engine": None,             # if None, keep scene's engine
        "samples": None,            # if None, keep scene's sample count
    }
    i = 0
    while i < len(argv):
        k = argv[i]
        if k == "--out-dir":
            out["out_dir"] = argv[i + 1]; i += 2
        elif k == "--num-frames":
            out["num_frames"] = int(argv[i + 1]); i += 2
        elif k == "--downscale":
            out["downscale"] = int(argv[i + 1]); i += 2
        elif k == "--camera":
            out["camera"] = argv[i + 1]; i += 2
        elif k == "--frame-start":
            out["frame_start"] = int(argv[i + 1]); i += 2
        elif k == "--engine":
            out["engine"] = argv[i + 1]; i += 2  # CYCLES | BLENDER_EEVEE | BLENDER_EEVEE_NEXT
        elif k == "--samples":
            out["samples"] = int(argv[i + 1]); i += 2
        else:
            raise ValueError(f"unknown arg: {k}")
    if out["out_dir"] is None:
        raise ValueError("--out-dir is required")
    return out


def configure_render(scene, *, downscale: int, engine: str | None, samples: int | None) -> None:
    """Set render settings while preserving the scene's aspect ratio.

    Output PNG is written via Image.save() in `render_frame`, NOT via
    scene.render.image_settings — see comment there.
    """
    if engine is not None:
        scene.render.engine = engine
    # Keep aspect ratio; just downscale.
    scene.render.resolution_x = max(1, scene.render.resolution_x // downscale)
    scene.render.resolution_y = max(1, scene.render.resolution_y // downscale)
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False  # required for clean per-frame poses
    scene.render.use_stamp = False  # no burn-in text on the image
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    if samples is not None:
        if scene.render.engine == "CYCLES":
            scene.cycles.samples = samples
        elif scene.render.engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
            scene.eevee.taa_render_samples = samples


def link_production_lights(scene) -> int:
    """If the scene has no LIGHT objects, pull lamps from each linked library.

    Sintel-era shot .blends often link in geometry/animation from environment
    .blend files but never explicitly link the *Light objects* — only the Light
    *datablocks* travel with the dependency graph. The result is that a fresh
    headless render produces a dark scene even though the production team
    authored a full lighting rig.

    This function looks at every library the scene already has loaded
    (bpy.data.libraries) and re-links each library's LIGHT objects into the
    active scene's collection. Returns the number of lights added. No-op if
    the scene already has at least one LIGHT object.
    """
    has_light = any(o.type == "LIGHT" for o in scene.objects)
    if has_light:
        return 0

    added = 0
    for lib in list(bpy.data.libraries):
        lib_path = bpy.path.abspath(lib.filepath)
        with bpy.data.libraries.load(lib_path, link=True) as (data_from, data_to):
            data_to.objects = [
                name for name in data_from.objects
                if "Lamp" in name or "Light" in name or "Sun" in name
            ]
        for obj in data_to.objects:
            if obj is None or obj.type != "LIGHT":
                continue
            scene.collection.objects.link(obj)
            added += 1
    return added


def render_frame(out_path: Path) -> None:
    """Render the active frame and write it as PNG.

    Two Blender-5.x quirks compounded:
    1. Sintel-era .blend files have `scene.render.image_settings.file_format` locked
       to FFMPEG. The rna's enum_items reports all 16 formats but runtime assignment
       fails: `enum "PNG" not found in ('FFMPEG')`. So we can't use the normal
       `bpy.ops.render.render(write_still=True)` path.
    2. Setting `Image.file_format = 'PNG'` on the Render Result image and calling
       `img.save()` works once, but the operation invalidates the Render Result so
       the next frame's render result is empty. Workaround: copy the Render Result
       to a throwaway image, save the copy, delete it.
    """
    bpy.ops.render.render(write_still=False)
    src = bpy.data.images["Render Result"]
    img = src.copy()
    img.filepath_raw = str(out_path)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def export_camera_state(scene, cam_obj) -> dict:
    """Read intrinsics + extrinsics for the active frame and return OpenCV-form."""
    cam = cam_obj.data
    intr = intrinsics_from_blender_camera(
        lens_mm=cam.lens,
        sensor_width_mm=cam.sensor_width,
        sensor_height_mm=cam.sensor_height,
        sensor_fit=cam.sensor_fit,
        shift_x=cam.shift_x,
        shift_y=cam.shift_y,
        pixel_aspect_x=scene.render.pixel_aspect_x,
        pixel_aspect_y=scene.render.pixel_aspect_y,
        resolution_x=scene.render.resolution_x,
        resolution_y=scene.render.resolution_y,
    )
    M_blender = np.array(cam_obj.matrix_world, dtype=np.float64)
    c2w_opencv = blender_world_matrix_to_opencv_c2w(M_blender)
    return {
        "K": intr.as_K().tolist(),
        "c2w_opencv": c2w_opencv.tolist(),
        "width": intr.width,
        "height": intr.height,
        "lens_mm": float(cam.lens),
        "sensor_width_mm": float(cam.sensor_width),
        "sensor_height_mm": float(cam.sensor_height),
        "sensor_fit": cam.sensor_fit,
    }


def main() -> None:
    args = parse_args(sys.argv)
    out_dir = Path(args["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    if args["camera"] is not None:
        cam_obj = bpy.data.objects[args["camera"]]
        scene.camera = cam_obj
    cam_obj = scene.camera
    if cam_obj is None:
        raise RuntimeError("scene has no active camera and --camera not specified")

    configure_render(
        scene,
        downscale=args["downscale"],
        engine=args["engine"],
        samples=args["samples"],
    )

    n_lights_added = link_production_lights(scene)
    if n_lights_added:
        print(f"[render_scene] linked {n_lights_added} production light objects "
              f"from {len(bpy.data.libraries)} libraries", flush=True)

    start = args["frame_start"] if args["frame_start"] is not None else scene.frame_start
    if args["num_frames"] is not None:
        end_excl = start + args["num_frames"]
    else:
        end_excl = scene.frame_end + 1  # render through scene.frame_end inclusive

    print(
        f"[render_scene] active camera object='{cam_obj.name}' "
        f"data='{cam_obj.data.name}' lens={cam_obj.data.lens}mm",
        flush=True,
    )
    print(
        f"[render_scene] engine={scene.render.engine} "
        f"resolution={scene.render.resolution_x}x{scene.render.resolution_y} "
        f"frames={start}..{end_excl - 1} ({end_excl - start} total)",
        flush=True,
    )

    frames_meta = []
    for i, f in enumerate(range(start, end_excl)):
        scene.frame_set(f)
        # matrix_world is animated; refresh after frame_set before reading.
        bpy.context.view_layer.update()

        frame_path = out_dir / f"frame_{i:04d}.png"
        render_frame(frame_path)

        cam_state = export_camera_state(scene, cam_obj)
        cam_state["frame_index"] = i
        cam_state["blender_frame"] = f
        cam_state["image_path"] = frame_path.name
        frames_meta.append(cam_state)
        print(f"[render_scene] frame {i:04d} (blender frame {f}) -> {frame_path.name}", flush=True)

    manifest = {
        "convention": "opencv",
        "camera_object_name": cam_obj.name,
        "camera_data_name": cam_obj.data.name,
        "scene_name": scene.name,
        "blender_version": bpy.app.version_string,
        "render_engine": scene.render.engine,
        "frames": frames_meta,
    }
    (out_dir / "cameras.json").write_text(json.dumps(manifest, indent=2))
    print(f"[render_scene] wrote cameras.json with {len(frames_meta)} frames", flush=True)


if __name__ == "__main__":
    main()
