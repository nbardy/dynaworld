# Blur / DoF / Motion Corpus Extraction Notes

Date: 2026-04-24

Scope:

Follow-up pass after building the local paper corpus in
`research_notes/blur_dof_motion_papers/`. The corpus currently has 55 indexed
papers, 55 PDFs, and 55 `pdftotext -layout` extracts.

## What Changed After Expanding The Search

The first review was directionally right, but the expanded corpus makes three
points sharper.

1. Dynamic blur is not a small extension of static camera blur. Recent 2025
   3DGS work treats dynamic-object motion, latent camera trajectories, dynamic
   Gaussian densification, and foreground remapping as core mechanisms.
2. Event/spike/IMU papers are mostly initialization and trajectory-supervision
   levers. They help severe blur, but they do not replace the need for a sharp
   world plus capture-state renderer.
3. Defocus papers keep converging on the same control split: focus distance /
   inverse focus distance plus CoC or aperture strength. Physical f-stop is a
   calibrated UI quantity unless sensor scale and metric scene scale are known.

Current belief:

The DynaWorld renderer should keep the exact-sampling path as the reference
model:

```text
I_obs ~= average_{shutter samples, aperture samples}
    render(W(t + tau), C(tau, aperture_sample), lens_state)
```

Covariance inflation, CoC convolution, and learned blur kernels can be fast
approximations, but only after matching the sampled reference on synthetic
tests.

## Extraction Anchors

These line anchors are from local text extracts. They are not polished paper
summaries; they are the places future agents should start when building the
formula/loss/dataset tables.

### Defocus / Depth Of Field

- `text/2405_17351_dof_gs_adjustable_depth_of_field_3d_gaussian_splatting.txt`
  - lines 72-83: DOF-GS motivation for finite-aperture camera and CoC-guided
    defocus rendering.
  - lines 146-167: focal length, aperture diameter, object depth, focal
    distance, and `Q = F A` relationship.
  - lines 283-287: training loop with per-view learnable `f_m`, `Q_m`, CoC map,
    depth map, and all-in-focus render.
- `text/2208_00945_dof_nerf_depth_of_field_meets_neural_radiance_fields.txt`
  - around lines 680-710: aperture parameter, focus distance, signed defocus
    map, and CoC kernel construction.
  - around lines 722-739: circular and polygonal CoC details.
  - around lines 899-900: controllable DoF by changing aperture size, focus
    distance, camera pose, perspective, and aperture shape.
- `text/2511_10316_depth_consistent_3d_gaussian_splatting_via_physical_de.txt`
  - lines 178-204: physical CoC model and conversion into pixel-space defocus.
  - lines 221-229: Gaussian/polygonal/SmoothStep kernel variants.
  - lines 332-334: focus-distance optimization from depth distribution.

Open extraction:

- Normalize all CoC formulas into one coordinate convention.
- Separate physical units (`F`, aperture diameter, sensor scale) from renderer
  units (`Q_eff`, pixel CoC radius).
- Identify which papers explicitly render an all-in-focus auxiliary image.

### Camera Motion Blur

- `text/2211_12853_bad_nerf_bundle_adjusted_deblur_neural_radiance_fields.txt`
  - lines 162-177: blurred image as virtual sharp images integrated over
    exposure time; optimize start/end poses and interpolate in `SE(3)`.
  - lines 195-213: sampled exposure-time pose construction.
  - lines 323-327: trajectory representation ablation; linear start/end poses
    were often sufficient for short exposure.
- `text/2404_11358_deblurgs_gaussian_splatting_for_camera_motion_blur.txt`
  - lines 233-236: camera motion blur as irradiance integration over
    time-varying 6-DoF pose `P_tau in SE(3)`.
  - lines 275-283: Bezier camera trajectory and sub-frame alignment.
  - around line 460: 9th-order Bezier curve used in the reported setup.
- `text/2503_05332_comogaussian_continuous_motion_aware_gaussian_splattin.txt`
  - lines 211-260: continuous camera trajectory with neural ODE latent features.
  - lines 318-333: aggregate renders along the camera trajectory into the
    blurry image.
  - lines 396-413: ODE-vs-MLP/GRU ablation.

Open extraction:

- Compare `T_start/T_end`, Bezier, spline, and neural ODE trajectories under one
  `ShutterState` schema.
- Build a synthetic test where true blur is non-linear; measure when linear
  constant-velocity shutter fails.
- Keep exposure duration explicit instead of hiding it in trajectory magnitude.

### Dynamic Object Blur

- `text/2403_10103_dyblurf_dynamic_neural_radiance_fields_from_blurry_mon.txt`
  - lines 502-508: DyBluRF conclusion: camera trajectory plus object-motion DCT
    trajectory through cross-time rendering.
  - lines 897-915: failure mode under extreme object motion; foreground
    information loss and dynamic-object recovery remain hard.
- `text/2510_10691_dynamic_gaussian_splatting_from_defocused_and_motion_b.txt`
  - lines 63-75: unified dynamic 3DGS framework for defocused and motion-blurred
    monocular videos.
  - lines 145-166: static/dynamic Gaussian split with compact `SE(3)` motion
    bases.
  - line 253: dynamic-object and scene depth correlation with blur magnitude.
- `text/2509_00831_upgs_unified_pose_aware_gaussian_splatting_for_dynamic.txt`
  - lines 88-108: dynamic scene deblurring framework and per-primitive `SE(3)`
    affine transformations for camera/object motion.

Open extraction:

- Decide whether DynaWorld's first implementation samples `W(t + tau)` through
  the existing time-conditioned generator, or adds a cheaper per-splat velocity
  approximation.
- Add a diagnostic that decomposes blur into background camera blur vs foreground
  object blur.
- Add a failure test where camera trajectory is allowed to explain a moving
  foreground object; the held-out view should expose the cheat.

### Event / Spike / IMU Assistance

Useful papers in the index:

- EvaGaussians, EaDeblur-GS, E-3DGS, DeblurSplat, BeSplat, PEGS, TRGS-SLAM,
  Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones.

Current model:

These papers are not the base product path unless the input device has those
sensors. The reusable idea is that high-temporal-resolution side channels can
initialize or regularize the shutter trajectory. They should enter as optional
evidence for `ShutterState`, not as a separate renderer.

## DynaWorld State Schema Draft

```text
CameraState:
    pose_mid: SE3
    fx, fy, cx, cy
    image_size
    crop_resize_transform
    distortion_optional

LensState:
    inv_focus_depth
    log_Q_eff
    aperture_shape
    calibrated_physical_F_optional
    calibrated_f_number_optional

ShutterState:
    t_mid
    exposure_duration
    shutter_curve
    trajectory_type: linear_se3 | bezier_se3 | spline_se3 | ode_latent
    trajectory_params
    rolling_shutter_readout_optional

DynamicState:
    sample_world_at(t + tau)
    optional per_splat_velocity_for_fast_approx
    optional foreground/object_motion_mask
```

## Immediate Next Extraction Pass

1. Build `formula_index.md` with exact equations and variable definitions from
   the A-priority papers.
2. Build `dataset_index.md` with which papers use synthetic blur, real blur,
   defocus datasets, event/spike sensors, dynamic monocular video, or SLAM.
3. Build `implementation_choices.md` that maps each paper to one of:
   exact sampling, CoC/covariance approximation, learned kernel, event-assisted
   trajectory supervision, or dynamic object-motion decomposition.
4. Add a synthetic renderer test plan before touching production renderer code.

