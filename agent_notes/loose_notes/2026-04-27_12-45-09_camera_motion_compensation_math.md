# Camera Motion Compensation Math

## Context

Nicholas asked for a defended mathematical argument about why increased camera
motion in the best V-JEPA static/dynamic run is a red flag, what is wrong or
not wrong with it, and what distributions/ranges make the argument valid.

This note is not claiming camera motion is bad. It formalizes why source-view
photometric training cannot by itself prove whether motion belongs to camera
state or dynamic scene state.

## Variables And Domains

Current tiny dog-clip setting:

```text
t_i                  normalized frame time in [0, 1]
I_i                  observed RGB frame in [0, 1]^(H x W x 3), H=W=128
K_i                  camera intrinsics, currently learned/derived from camera head
C_i in SE(3)         camera pose or world-to-camera transform
S                    static Gaussian bank
D_i                  dynamic Gaussian bank evaluated at time t_i
R(K_i, C_i, G_i)     differentiable Gaussian renderer
G_i = S union D_i    full time-conditioned splat set
```

Prediction:

```text
Ihat_i = R(K_i, C_i, S union D_i)
```

Training objective, simplified:

```text
L = E_i[ photometric(Ihat_i, I_i) ]
  + lambda_cam_motion * L_cam_motion(C_1:T)
  + lambda_cam_temporal * L_cam_temporal(C_1:T)
  + lambda_cam_global * L_cam_global(C_1:T)
  + lambda_dyn * L_dyn(D_1:T)
```

For the current configs, the source distribution is narrow:

```text
D_source:
    one small video clip
    16 loaded/training frames for the strongest static/dynamic V-JEPA recipe
    source-view photometric supervision
    no independent ground-truth camera path
    no same-time held-out camera view
```

Camera/dynamic ranges from the best-recipe configs:

```text
camera.max_rotation_degrees       5.0
camera.max_translation_ratio      0.2
camera.base_radius                3.0
dynamic_motion_extent             0.375
dynamic_time_basis_count          8
dynamic_rotation_degrees          10.0
dynamic_alpha_logit_extent        2.0
```

Observed adjacent camera rotation summaries:

```text
local static/dynamic              about 0.0159 deg/frame
V-JEPA static/dynamic 250 step     about 0.1309 deg/frame
V-JEPA static/dynamic ~525 step    about 0.1827 deg/frame
```

These are still much smaller than the configured 5 degree cap. The issue is not
"too large to be physically possible." The issue is attribution.

## Claim

Claim:

```text
In source-view photometric training, camera motion and scene motion are
partially non-identifiable. An improved source-view loss with larger learned
camera motion does not prove the dynamic splats learned the real object motion.
```

Confidence:

```text
high for source-view photometric loss
medium for this exact run, because the dynamic metrics also increased
```

Important non-claim:

```text
Increasing camera motion is not inherently bad. If the video has real camera
motion, the correct camera estimate should move. The red flag is only that the
same loss can reward wrong attribution.
```

## Local Linearized Geometry

For a 3D point `X_i` rendered at time `i`:

```text
x_i = C_i X_i                         camera-frame point
u_i = pi(K_i x_i)                     image pixel/projection
```

For small perturbations:

```text
delta u ~= J_X delta X + J_xi delta xi
```

Where:

```text
delta X       scene/object motion in world coordinates
delta xi      camera twist in se(3): 3 rotation + 3 translation parameters
J_X           image Jacobian wrt object position
J_xi          image Jacobian wrt camera motion
```

Photometric loss observes image residuals, not latent causes:

```text
photometric residual -> desired delta u
```

But the equation:

```text
delta u = J_X delta X + J_xi delta xi
```

usually has many solutions. For any camera-motion choice `delta xi`, a different
scene-motion choice `delta X` can produce nearly the same pixel displacement in
the source view, especially when only one camera view supervises each time.

Therefore source-view loss alone cannot uniquely decide:

```text
object moved right
```

versus:

```text
camera panned left
```

or any mixture of both.

## Simple 1D Projection Example

Use a pinhole camera with focal length `f`, point depth `z`, lateral coordinate
`x`, and image coordinate:

```text
u = f * x / z
```

Object lateral motion:

```text
x' = x + delta_x
delta u_object ~= f * delta_x / z
```

Camera lateral translation by `tau_x` changes camera-frame point approximately:

```text
x_c' = x - tau_x
delta u_camera ~= -f * tau_x / z
```

For a single source view:

```text
f * delta_x / z ~= -f * tau_x / z
```

so:

```text
delta_x ~= -tau_x
```

Both explanations can produce the same pixel shift. Multi-view data, known
cameras, or independent motion priors are needed to separate them robustly.

## Why Static/Dynamic V-JEPA Makes This More Subtle

The best run did not only increase camera motion. It also increased decoded
dynamic motion:

```text
local static/dynamic decoded XYZ adjacent     about 0.0455
V-JEPA static/dynamic 250 decoded XYZ adjacent about 0.0945
V-JEPA static/dynamic ~525 decoded XYZ adjacent about 0.1305
```

That weakens the overly simple criticism "it is only camera." The better
statement is:

```text
The run uses both dynamic splat motion and more camera motion. We do not yet
know whether the camera motion is correct camera recovery, useful nuisance
alignment, or a substitute for some missing scene motion.
```

## Distribution Argument

The concern depends on train/eval distribution.

### Source-View Same-Clip Distribution

```text
D_source = {same camera/view, same clip, same train/eval frame family}
```

Under `D_source`, camera compensation can be rewarded because the only target is
the rendered source image:

```text
minimize E_{(I_t,t)~D_source} photometric(R(C_t, S, D_t), I_t)
```

Wrong 3D decomposition can still render the source view well.

### Held-Out Same-Scene Camera Distribution

```text
D_multiview = {(I_{t,c}, t, camera c) for multiple cameras at same time}
```

Under `D_multiview`, a wrong camera/scene decomposition is harder to hide:

```text
R(C_{t,c1}, S, D_t) and R(C_{t,c2}, S, D_t)
```

must both match. If object motion was actually encoded as source-camera motion,
the held-out camera render should fail.

### Scene-Distinct Clip Distribution

```text
D_scene = many clips/scenes with different camera/object/background motion
```

Under `D_scene`, memorized camera tricks should generalize poorly unless the
camera head learned a reusable camera-motion estimator.

## Proposed Camera-Clamp Control

Do not replace the flexible-camera baseline. Add a sibling.

Baseline:

```text
camera.max_rotation_degrees      = 5.0
camera.max_translation_ratio     = 0.2
loss.camera_motion_weight        = 0.01
loss.camera_temporal_weight      = 0.02
```

Clamp/control idea:

```text
camera.max_rotation_degrees      in {0.5, 1.0, 2.0}
camera.max_translation_ratio     in {0.02, 0.05, 0.1}
loss.camera_motion_weight        in {0.05, 0.1}
loss.camera_temporal_weight      in {0.05, 0.1}
```

Start with one moderate control, not a sweep:

```text
max_rotation_degrees  = 1.0
max_translation_ratio = 0.05
camera_motion_weight  = 0.05
camera_temporal_weight = 0.05
```

Reason:

```text
It should reduce large camera-path freedom without making the camera exactly
static. Exact-static camera would test a different, harsher hypothesis.
```

## Metrics

Use dimensionless or comparable quantities:

```text
Eval/Loss
Eval/SSIM
Eval/TemporalAdjacentL1Ratio
Eval/DecodedXYZAdjacentL2
Camera/EvalAdjacentRotationDeltaDegrees
Camera/EvalAdjacentTranslationDelta
BankRate/dynamic_motion
motion-masked L1 when available
high-pass or edge L1 when available
```

Useful ratios:

```text
R_img = pred_adjacent_L1 / gt_adjacent_L1
R_cam = camera_adjacent_rotation_deg / max_rotation_degrees
R_dyn = decoded_xyz_adjacent_L2 / scene_extent
```

## Pseudocode Test

```python
configs = {
    "free_camera": base_vjepa_static_dynamic_config,
    "camera_clamped": {
        **base_vjepa_static_dynamic_config,
        "camera.max_rotation_degrees": 1.0,
        "camera.max_translation_ratio": 0.05,
        "losses.camera_motion_weight": 0.05,
        "losses.camera_temporal_weight": 0.05,
    },
}

for name, cfg in configs.items():
    run = train(cfg, steps=matched_steps, seed=matched_seed)
    metrics[name] = evaluate(run, frames="full_video")

def read(metrics):
    free = metrics["free_camera"]
    clamp = metrics["camera_clamped"]

    if close(clamp.loss, free.loss) and clamp.camera_motion < free.camera_motion:
        if close_or_higher(clamp.dynamic_motion, free.dynamic_motion):
            return "dynamic splats are likely carrying the motion; camera freedom was not essential"
        return "source-view fit may not require real motion; inspect visual/motion masks"

    if clamp.loss > free.loss and clamp.temporal_adjacent_ratio < free.temporal_adjacent_ratio:
        return "free-camera run probably used camera freedom for useful alignment or compensation"

    if clamp.loss > free.loss and clamp.dynamic_motion > free.dynamic_motion:
        return "dynamic bank tried to compensate but lacked capacity/conditioning"

    return "ambiguous; use held-out camera or motion-masked metrics"
```

## Outcome Table

| Outcome | Interpretation | Next action |
| --- | --- | --- |
| Loss/SSIM stay strong, camera motion drops, dynamic motion stays high | Camera increase was not essential; dynamic representation is credible | Keep flexible baseline, prefer clamped as cleaner evidence |
| Loss/SSIM collapse, camera motion drops, dynamic motion also drops | Model relied on camera freedom and did not learn enough scene dynamics | Need stronger dynamic bank/loss/sampler or known-camera/multiview supervision |
| Loss/SSIM collapse, dynamic motion rises | Dynamic bank has signal but cannot replace camera path under current capacity | Tune dynamic capacity before camera claims |
| Loss/SSIM stay strong, both camera and dynamic motion drop | Same-source task may be too easy/static; metrics not measuring object motion | Add motion masks/high-pass/held-out views |
| Flexible and clamped both fail held-out camera | Source-view overfit is not world-consistent | Move to multiview/scene-distinct contract |

## Decision

Current recommendation:

```text
Do not remove camera flexibility from the best recipe.
Do add a camera-clamped sibling as a falsification test.
Treat increased camera motion as a warning label, not as a failure.
```

The best statement to make right now:

```text
The V-JEPA static/dynamic recipe improved source-view fit while increasing both
dynamic splat motion and learned camera motion. This is promising, but source-
view photometric loss cannot prove the attribution. A camera-clamped sibling and
held-out/multiview validation are the correct tests.
```
