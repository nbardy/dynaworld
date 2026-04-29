"""Runs INSIDE Blender 2.79b. Adaptation of `render_scene.py` for the older API.

Why a separate script: Sintel was authored for Blender Internal renderer
(`BLENDER_RENDER`), removed from Blender in 2.80. Modern Blender (5.1) renders
the same .blend with broken materials (pink-blob output). Blender 2.79b is the
last version that natively renders Sintel's shader graphs faithfully.

API differences vs the modern script:
- LAMP object type instead of LIGHT
- `scene.update()` instead of `view_layer.update()` after `frame_set`
- bpy.app.version is (2, 79, 0)

Camera + intrinsics math is unchanged: `Camera.lens`, `sensor_*`, `shift_*`,
`Object.matrix_world` have been stable since Blender 2.5x.
"""
from __future__ import absolute_import, print_function

import json
import os
import sys

import bpy

# camera_export.py is sibling — same 2-line numpy module that works on any Python.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
from camera_export import (  # noqa: E402
    blender_world_matrix_to_opencv_c2w,
    intrinsics_from_blender_camera,
)

import numpy as np  # noqa: E402  (Blender 2.79b ships with numpy)


def parse_args(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    out = {
        "out_dir": None,
        "num_frames": None,
        "frame_start": None,
        "downscale": 8,
        "camera": None,
        "samples": None,
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
        elif k == "--samples":
            out["samples"] = int(argv[i + 1]); i += 2
        else:
            raise ValueError("unknown arg: %s" % k)
    if out["out_dir"] is None:
        raise ValueError("--out-dir is required")
    return out


def link_production_lights(scene):
    """Link LAMP objects from already-loaded libraries into the active scene.

    See render_scene.py for the why; same shape, different object-type name in 2.79b.
    """
    has_lamp = any(o.type == "LAMP" for o in scene.objects)
    if has_lamp:
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
            if obj is None or obj.type != "LAMP":
                continue
            scene.objects.link(obj)  # 2.79b API: scene.objects.link, not scene.collection.objects.link
            # Place lamps on all layers so they're guaranteed to be in the render
            # regardless of which scene layer the user enables.
            obj.layers = [True] * 20
            added += 1
    return added


def configure_render(scene, downscale, samples):
    """Preserve aspect ratio at downscale, force PNG output, enable all layers.

    The .blend ships with only L01 enabled in scene.layers; the camera and most
    of the geometry live on L09 along with various props on L00/L02/L11/L14/L18.
    Enabling all 20 layers is the headless-rendering equivalent of clicking
    every layer button in 2.79b's UI before hitting render.
    """
    scene.render.resolution_x = max(1, scene.render.resolution_x // downscale)
    scene.render.resolution_y = max(1, scene.render.resolution_y // downscale)
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False
    scene.render.use_stamp = False
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.layers = [True] * 20
    try:
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        return "PNG_DIRECT"
    except TypeError:
        # 2.79b locks image_settings same way 5.1 does — fall back at save time.
        return "PNG_VIA_IMAGE_SAVE"
    return None


def render_frame(out_path, save_mode):
    if save_mode == "PNG_DIRECT":
        bpy.context.scene.render.filepath = str(out_path).rsplit(".png", 1)[0]
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(write_still=False)
        src = bpy.data.images["Render Result"]
        img = src.copy()
        img.filepath_raw = str(out_path)
        img.file_format = "PNG"
        img.save()
        bpy.data.images.remove(img)


def export_camera_state(scene, cam_obj):
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


def main():
    args = parse_args(sys.argv)
    out_dir = args["out_dir"]
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    scene = bpy.context.scene
    if args["camera"] is not None:
        cam_obj = bpy.data.objects[args["camera"]]
        scene.camera = cam_obj
    cam_obj = scene.camera
    if cam_obj is None:
        raise RuntimeError("scene has no active camera and --camera not specified")

    save_mode = configure_render(scene, args["downscale"], args["samples"])
    print("[render_scene_279] save_mode=%s" % save_mode, flush=True)

    n_lights = link_production_lights(scene)
    if n_lights:
        print("[render_scene_279] linked %d production lamps from %d libs" % (
            n_lights, len(bpy.data.libraries)), flush=True)

    start = args["frame_start"] if args["frame_start"] is not None else scene.frame_start
    if args["num_frames"] is not None:
        end_excl = start + args["num_frames"]
    else:
        end_excl = scene.frame_end + 1

    print("[render_scene_279] active camera object='%s' data='%s' lens=%smm" % (
        cam_obj.name, cam_obj.data.name, cam_obj.data.lens), flush=True)
    print("[render_scene_279] engine=%s resolution=%dx%d frames=%d..%d (%d total)" % (
        scene.render.engine, scene.render.resolution_x, scene.render.resolution_y,
        start, end_excl - 1, end_excl - start), flush=True)

    frames_meta = []
    for i, f in enumerate(range(start, end_excl)):
        scene.frame_set(f)
        scene.update()  # 2.79b equivalent of view_layer.update

        frame_path = os.path.join(out_dir, "frame_%04d.png" % i)
        render_frame(frame_path, save_mode)

        cam_state = export_camera_state(scene, cam_obj)
        cam_state["frame_index"] = i
        cam_state["blender_frame"] = f
        cam_state["image_path"] = "frame_%04d.png" % i
        frames_meta.append(cam_state)
        print("[render_scene_279] frame %04d (blender frame %d)" % (i, f), flush=True)

    manifest = {
        "convention": "opencv",
        "camera_object_name": cam_obj.name,
        "camera_data_name": cam_obj.data.name,
        "scene_name": scene.name,
        "blender_version": bpy.app.version_string,
        "render_engine": scene.render.engine,
        "frames": frames_meta,
    }
    with open(os.path.join(out_dir, "cameras.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)
    print("[render_scene_279] wrote cameras.json with %d frames" % len(frames_meta), flush=True)


if __name__ == "__main__":
    main()
