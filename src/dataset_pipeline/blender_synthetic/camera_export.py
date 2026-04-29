"""Blender camera -> OpenCV `c2w` + pixel-space K extraction.

Conventions:
- Blender camera frame: -Z forward, +Y up, +X right.
- OpenCV / 3DGS:        +Z forward, -Y up, +X right.

The flip is in *camera space* (right-multiply the world matrix). Easy to get
wrong; this module is the single source of truth and never reimplements it
inline.

Pixel-space K from Blender requires branching on `sensor_fit`. Reference:
BlenderProc CameraUtility.

Compatibility: written in Python-3.5-friendly style (no `from __future__ import
annotations`, no parameterized type aliases, no dataclasses) so this same module
loads under Blender 2.79b's bundled Python 3.5 AND modern Python 3.10+.
"""
import numpy as np


# Camera-space axis flip: Blender -> OpenCV (flip Y and Z of the camera frame).
FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])


class CameraIntrinsics(object):
    """Pinhole intrinsics in pixel space, OpenCV convention.

    Plain class instead of @dataclass so this loads on Python 3.5 (Blender 2.79b).
    """

    __slots__ = ("fx", "fy", "cx", "cy", "width", "height")

    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = int(width)
        self.height = int(height)

    def as_K(self):
        K = np.eye(3)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K


def blender_world_matrix_to_opencv_c2w(M_blender):
    """Convert a Blender camera world matrix to OpenCV-convention c2w.

    Blender's `Object.matrix_world` puts the camera looking down -Z; OpenCV
    looks down +Z. We right-multiply by `diag(1,-1,-1,1)` to flip Y and Z of
    the *camera* frame while leaving the world frame untouched.
    """
    M = np.asarray(M_blender, dtype=np.float64).reshape(4, 4)
    return M.dot(FLIP_YZ)


def intrinsics_from_blender_camera(
    lens_mm,
    sensor_width_mm,
    sensor_height_mm,
    sensor_fit,
    shift_x,
    shift_y,
    pixel_aspect_x,
    pixel_aspect_y,
    resolution_x,
    resolution_y,
):
    """Compute pixel-space K from a Blender camera + scene render settings.

    Mirrors BlenderProc's CameraUtility logic. The `sensor_fit` field controls
    which sensor dimension drives the focal scale; the AUTO branch picks the
    larger pixel-aspect-weighted dimension.
    """
    w = int(resolution_x)
    h = int(resolution_y)
    par_x = float(pixel_aspect_x)
    par_y = float(pixel_aspect_y)

    if sensor_fit == "VERTICAL" or (
        sensor_fit == "AUTO" and h * par_y > w * par_x
    ):
        view_fac_px = par_y * h
        sensor_mm = sensor_height_mm
    else:
        view_fac_px = par_x * w
        sensor_mm = sensor_width_mm

    fx = (lens_mm / sensor_mm) * view_fac_px
    fy = fx * (par_y / par_x)
    cx = (w - 1) * 0.5 - shift_x * view_fac_px
    cy = (h - 1) * 0.5 + shift_y * view_fac_px / (par_x / par_y)

    return CameraIntrinsics(fx, fy, cx, cy, w, h)


def project_world_point_opencv(p_world, c2w, K):
    """Project a 3D world point through an OpenCV-convention c2w + K.

    Returns (u, v, z_camera). z_camera > 0 means in front of the camera.

    Convention: row-vector world-to-camera transform via
        p_cam_row = (p_world - center)_row @ c2w[:3,:3]
    The matrix named `c2w` acts as w2c when right-multiplied to a row vector.
    """
    p_world = np.asarray(p_world, dtype=np.float64).reshape(3)
    R = c2w[:3, :3]
    center = c2w[:3, 3]
    p_cam = (p_world - center).dot(R)
    z = p_cam[2]
    u = K[0, 0] * (p_cam[0] / z) + K[0, 2]
    v = K[1, 1] * (p_cam[1] / z) + K[1, 2]
    return float(u), float(v), float(z)
