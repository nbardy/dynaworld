# Dynamic PowerFoam Implicit Camera Wiring

## Context

The full-clip dynamic feature PowerFoam baseline used fixed pinhole rays. That
is the wrong factorization for high camera-motion clips: a mostly static scene
forces camera motion into moving foam cells, changing density/support, or
repainting features.

## Implementation Result

Added a scoped implicit-camera path for `train_dynamic_powerfoam_metal.py`.

Files:

```text
src/train/powerfoam_implicit_camera.py
src/train/train_dynamic_powerfoam_metal.py
tests/test_powerfoam_implicit_camera.py
tests/test_dynamic_powerfoam_metal.py
src/train_configs/local_mac_token_dynamic_powerfoam_features_F32_1024_8sites_youtube_hlaZbH_center_crop_8fps_512_56f_120step_implicit_camera.jsonc
```

The new camera branch is config-gated:

```jsonc
"camera": {
  "enabled": true,
  "mode": "learned_implicit"
}
```

Default dynamic PowerFoam remains fixed pinhole rays. The learned branch
initializes a base camera at `[0, 0, -base_radius]` looking at the origin, with
fixed intrinsics/lens and small learned SE(3) global/path residuals.

## Important Backtrack

The first naive integration tried to feed learned full rays into the existing
Metal renderer. A gradient probe showed that the current dynamic PowerFoam
Metal op does not return gradients to ray origins/directions:

```text
start_token_grad None
time_tokens_grad None
global_head_last_weight_grad None
offset_head_last_weight_grad None
offset_head_last_bias_grad None
```

So learned rays render but do not train the camera.

The working integration instead uses the equivalent camera-space transform for
the pinhole fixed-intrinsics case:

```text
world foam state + learned camera_to_world(t)
-> Torch world_to_camera(t) transform of points/normals/tangents
-> existing fixed-origin PowerFoam rays
-> Metal renderer
```

This lets gradients flow through the renderer's point/normal gradients back
into the learned camera pose.

## Verification

Focused tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train uv run --with pytest python \
  -m pytest -p no:cacheprovider \
  tests/test_powerfoam_implicit_camera.py tests/test_dynamic_powerfoam_metal.py -q -rs
```

Result:

```text
20 passed in 9.99s
```

The new MPS regression proves camera-pose gradients reach both global and
offset camera heads through the camera-space transform path.

One-step real trainer smoke:

```text
output_dir outputs/dynamic_powerfoam_metal/implicit_camera_smoke_32px_3f_1step
dynamic_mode token_rbf_features
camera_mode learned_implicit
render_size 32
frames 3
cells 32
steps 1
```

Result:

```text
step 1 eval_l1 0.485397
state_camera_rotation_delta_mean_degrees 0.289997
state_camera_translation_delta_mean 0.027190
state_camera_origin_delta_mean 0.027190
state_camera_forward_delta_mean 0.004133
```

This proves the training loop can move the learned camera. It is not a quality
claim and not a camera-recovery claim.

## Config Update

The full-clip implicit-camera config now uses random object-centric foam:

```text
init_from_video false
xy_extent 1.25
z_min -1.25
z_max 1.25
camera.base_radius 3.0
```

That matches the intended origin-looking camera initialization. A future
quality run may need an object-centric image/backprojection initializer, but
using the old positive-z image init would mix the fixed-ray coordinate
assumption into the implicit-camera lane.

## Open Risks

- The camera branch is source-view only until we add synthetic known-camera or
  heldout-camera controls.
- Learned camera and dynamic foam are still underidentified under monocular
  photometric loss.
- Camera-space rendering currently assumes fixed pinhole intrinsics. If we want
  learned fisheye/intrinsics, the renderer needs ray-gradient support or a
  differentiable lens-aware camera-space projection strategy.
- Existing temporal screen-motion metrics still assume fixed camera and should
  not be used alone to judge motion-vs-camera separation.
