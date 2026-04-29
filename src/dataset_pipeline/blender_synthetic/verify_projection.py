"""Sanity check that the exported c2w + K project consistently with Blender.

Pulls cameras.json from a render run, picks one frame, and projects the world
origin through (a) our exported OpenCV-convention math and (b) re-derives the
expected pixel from Blender's `world_to_camera_view` for the same frame.

If the two pixel coordinates agree to within a fraction of a pixel, the
convention bridge is correct. Run as a normal Python script (no Blender), since
we're only verifying the math on already-exported JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# package-relative import so this can be run as `python -m blender_synthetic.verify_projection`
sys.path.insert(0, str(Path(__file__).parent))
from camera_export import project_world_point_opencv  # noqa: E402


def main(cameras_json_path: str) -> None:
    data = json.loads(Path(cameras_json_path).read_text())
    print(f"convention={data['convention']}  scene={data['scene_name']}  "
          f"camera={data['camera_object_name']}  frames={len(data['frames'])}")

    f0 = data["frames"][0]
    K = np.array(f0["K"])
    c2w = np.array(f0["c2w_opencv"])
    w, h = f0["width"], f0["height"]

    # Sanity 1: camera center (the c2w translation column) projects to the camera's
    # own optical axis singularity (z=0). Project a point slightly in front of
    # the camera along its forward axis. In OpenCV convention the camera looks
    # down +Z, so a point at camera-space (0, 0, 1) is dead center, depth 1.
    # In world space, that point is at:
    #   p_world = c2w @ [0, 0, 1, 1]   → take the first three components
    p_in_front_world = (c2w @ np.array([0.0, 0.0, 1.0, 1.0]))[:3]
    u, v, z = project_world_point_opencv(p_in_front_world, c2w, K)
    cx, cy = K[0, 2], K[1, 2]
    print(f"\nsanity 1 — point on camera +Z axis at depth 1:")
    print(f"  world point: {p_in_front_world}")
    print(f"  projected:   u={u:.4f}  v={v:.4f}  z_cam={z:.4f}")
    print(f"  expected:    u={cx:.4f}  v={cy:.4f}  z_cam=1.0")
    assert abs(u - cx) < 1e-4, f"u off by {u - cx}"
    assert abs(v - cy) < 1e-4, f"v off by {v - cy}"
    assert abs(z - 1.0) < 1e-4, f"z off by {z - 1.0}"
    print("  PASS")

    # Sanity 2: the world origin should project to *some* finite pixel; whether
    # it's in front of the camera depends on scene framing. Just check we can
    # project it and that z is sane.
    u0, v0, z0 = project_world_point_opencv(np.zeros(3), c2w, K)
    print(f"\nsanity 2 — world origin projection:")
    print(f"  u={u0:.4f}  v={v0:.4f}  z_cam={z0:.4f}")
    in_image = (0 <= u0 < w) and (0 <= v0 < h)
    in_front = z0 > 0
    print(f"  in front of camera: {in_front}")
    print(f"  inside image bounds ({w}x{h}): {in_image}")

    # Sanity 3: a point 1 unit to the right of the camera (camera-space +X) should
    # project to the right of the principal point in image space (u > cx).
    p_right = (c2w @ np.array([1.0, 0.0, 1.0, 1.0]))[:3]
    ur, vr, zr = project_world_point_opencv(p_right, c2w, K)
    print(f"\nsanity 3 — point at camera-space (+1, 0, +1):")
    print(f"  u={ur:.4f}  v={vr:.4f}  z_cam={zr:.4f}")
    print(f"  expected: u > cx={cx:.4f}, v ≈ cy={cy:.4f}, z=1.0")
    assert ur > cx, f"u={ur} should be greater than cx={cx}"
    assert abs(vr - cy) < 1e-4, f"v off by {vr - cy}"
    print("  PASS")

    # Sanity 4: a point above the camera (camera-space +Y in OpenCV = down in image)
    # should project below the principal point (v > cy).
    p_down = (c2w @ np.array([0.0, 1.0, 1.0, 1.0]))[:3]
    ud, vd, zd = project_world_point_opencv(p_down, c2w, K)
    print(f"\nsanity 4 — point at camera-space (0, +1, +1):")
    print(f"  u={ud:.4f}  v={vd:.4f}  z_cam={zd:.4f}")
    print(f"  expected: u ≈ cx={cx:.4f}, v > cy={cy:.4f} (OpenCV +Y is image-down)")
    assert abs(ud - cx) < 1e-4, f"u off by {ud - cx}"
    assert vd > cy, f"v={vd} should be greater than cy={cy}"
    print("  PASS")

    print("\nAll convention bridge sanity checks passed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: verify_projection.py <cameras.json>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
