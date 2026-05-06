# PowerFoam Math Implementation Hooks

Date: 2026-05-05 23:28:14 +0700

Scope: implementation-facing note only. No code changes are made here. This
assumes the current lane boundary: local PowerFoam Metal forward/backward and
4K gates are useful, official CUDA/Warp fixture is still absent on this Mac,
and paper acceptance is blocked by heldout quality rather than by a missing
local smoke gate.

## Current PowerFoam Metal Path

Primary runtime path:

```text
src/train/train_powerfoam_metal.py
    resolve_config(...)
    MetalPowerFoamVideo
        raw parameters -> decoded_parameters / decoded_texel_* / decoded_texel_sv
        build_csr_adjacency(points[frame], radii[frame], mode=adjacency_mode)
        rasterize_power_foam_* or raytrace_power_foam_*
    training loop
        RGB losses + optional normal/contribution/interpenetration terms
        log_artifacts / aux_metrics / checkpoint_best.pt / checkpoint_final.pt

third_party/powerfoam-metal/torch_powerfoam_metal/rasterize.py
    Python shape checks, feature packing, custom autograd Functions
    calls torch.ops.powerfoam_metal.* forward/backward

third_party/powerfoam-metal/csrc/{bindings.cpp,shared/common.h,metal/*.mm,metal/*.metal}
    op schema, MPS tensor checks, kernel launch arg order, Metal math
```

The full current primitive is `feature_mode == "quaternion_height_sv_texel_surface"`.
That is the best hook target because it uses the current paper-adjacent
parameters:

```text
points / centers       [T,N,3] decoded from raw_xy/raw_z
radii                  [T,N]   softplus(raw_radii) + radius_min
densities              [T,N]   softplus(raw_densities)
quaternions            [T,N,4] raw quaternion frame, normalized in quaternion_frames
texel_sites            [T,N,S,2] local tangent chart coordinates
texel_height           [T,N,S] world-height displacement, radius-scaled in decode
texel_sv_axis          [T,N,S,D,3] SV color query axes
texel_sv_rgb           [T,N,S,D,3] SV RGB values, shifted by +0.5 in color eval
adjacency/offsets      CSR over selected cell neighbors
rays                   [B,H,W,6] origin + direction
```

At the wrapper boundary these become one `features_flat` tensor per frame:

```text
per texel stride = 2 site + 1 height + 3*D sv_axis + 3*D sv_rgb
per cell feature = S * (3 + 6D) + 9 frame values
frame values     = normal[3], tangent[3], bitangent[3]
```

Metal backward returns only:

```text
grad_points, grad_radii, grad_densities, grad_features
```

Everything else must be expressible through `features_flat` and PyTorch's
packing graph, or the op schema must grow. This is the central implementation
constraint for new math.

## Hook Classes

### A. Decode-Only Math

Use this if the idea can be represented as a deterministic transform before
the existing Metal op:

```text
raw params -> decoded centers/radii/quaternions/texels/SV -> existing raster op
```

Good fits:

- radius/height reparameterizations that still emit `radii` and `texel_height`
- quaternion/frame constraints that still emit `quaternions` or frame vectors
- transport priors over `texel_sites`, `texel_height`, `texel_sv_axis`,
  `texel_sv_rgb` that are computed in PyTorch outside the kernel
- temporal smoothness or material-freeze losses over decoded tensors

Contracts:

- no Metal forward change
- no Metal backward change
- add config under `model` or `losses`, normalized in `resolve_config`
- add a scheduled loss term in the trainer if it affects optimization
- log scalar diagnostics in `train_metrics_history.jsonl` and W&B

This is the preferred first insertion point for gauge/fluid math because it
does not risk breaking the local 4K Metal core.

### B. Ray-Incidence Math

Use this if the math changes optical depth, interval clipping, cell ownership,
or texel sampling along the ray.

Concrete Metal seams:

```text
powerfoam_tiled_stream_kernels.metal
    stream_clipped_cell_interval_diff(...)
    stream_clip_height_surface(...)
    stream_sv_texel_color(...)
    powerfoam_tiled_forward
    powerfoam_tiled_backward_global_atomic
    powerfoam_tiled_backward_height_sv_feature_reduced
    powerfoam_raytrace_forward_height_sv
    powerfoam_raytrace_backward_height_sv_global_atomic
```

Concrete wrapper seams:

```text
rasterize_power_foam_oriented_height_sv_texel_surface(...)
rasterize_power_foam_quaternion_height_sv_texel_surface(...)
raytrace_power_foam_oriented_height_sv_texel_surface(...)
FoamRasterConfig
_make_meta(... feature_mode, sv_dof, depth_quantile_count)
```

Contracts:

- tiled forward and raytrace forward must match for the selected mode
- backward must route gradients to all changed continuous inputs
- if a new output participates in loss, the custom autograd Function needs a
  `grad_<output>` argument and the C++/Metal op schema needs the same arg
- if the math introduces a new continuous primitive tensor, the op needs a new
  forward arg and a new backward return, unless it can be packed into
  `features_flat`
- any topology selection remains piecewise constant unless explicitly made
  differentiable; do not start by differentiating neighbor selection

### C. Aux/Diagnostic Math

Use this if the idea only needs evidence, not gradients:

```text
rasterize_power_foam_*_aux(...)
FoamAuxOutputs(
    normal_distance,
    normal,
    median_depth,
    contrib,
    point_error,
    visible_mask,
    depth_quantile_depths,
    depth_quantile_values,
)
MetalPowerFoamVideo.aux_metrics(...)
log_artifacts(...)
best_metrics.json / eval_metrics_history.jsonl
```

Good fits:

- face witness / cell witness metrics
- heldout residual by low-witness cell
- holonomy summaries over active adjacency
- contribution-weighted material transport error
- depth-quantile coverage and median-depth stability

Contracts:

- diagnostics that do not need gradients can run under `@torch.no_grad()`
- prefer adding aux outputs only when the kernel already computes the event
  cheaply; otherwise write an offline replay script under
  `research_experiments/dynamic_foam/`
- selector must remain heldout PSNR/SSIM/L1 or heldout residual diagnostics,
  not source-view fit

## Likely New Math Parameters

Do not add all of these. Pick the smallest parameter that falsifies the
hypothesis.

```text
connection/gauge:
    per-edge rest transport       [T,E,2,2] or implicit from quaternions
    per-cell chart scale/shear    [T,N,2] or [T,N,2,2]
    face witness cache           diagnostic only, not trainable

fluid/material:
    material_id logits            [T,N,K] or [N,K]
    texel velocity                [T,N,S,2] local chart flow
    height velocity               [T,N,S]
    SV axis transport residual    derived from texel_sv_axis, not first-class
    density continuity weight     scalar config, no new tensor

incidence:
    cell thickness / viscosity    [T,N] or packed feature channel
    texel normal correction       [T,N,S,3] if height gradients become explicit
    per-cell ray falloff params   [T,N,P] only if current density/radius fails
```

If a value is trainable and sampled in Metal per ray event, prefer packing it
into `features_flat` only when it is cell-local or texel-local and its gradient
can naturally live in `grad_features`. Add a first-class tensor only when it
has a different shape, lifecycle, LR group, or checkpoint meaning.

## Loss And Regularizer Hooks

Current loss surface already has scheduled weights for:

```text
l1, mse, ssim, radius_l2, density_l2,
normal_distance, contribution, interpenetration
```

New losses should follow the same pattern:

```text
LOSS_DEFAULTS:
    "<name>_weight": 0.0
    "<name>_weight_start_step": 0
    "<name>_weight_final_multiplier": 1.0

scheduled_loss_weights:
    "<name>_weight": aux_weight("<name>")

training loop:
    <name>_loss = helper(model, frame_indices, ...) if weight > 0 else zero
    loss += weight * <name>_loss
    train_metrics["<name>_loss"] = ...
    train_metrics["<name>_weight"] = ...
```

Candidate terms:

- `connection_loss`: contribution-weighted SO2/SO3 frame disagreement across
  active cell edges; first version should be stopgrad on edge weights.
- `texel_transport_loss`: compare material/SV values after transporting local
  texel coordinates across time or neighbor faces.
- `density_continuity_loss`: penalize density jumps across witnessed faces,
  not all Cech/AABB edges.
- `height_smoothness_loss`: local chart height Laplacian over texel sites;
  should be radius-normalized.
- `low_witness_suppression_loss`: penalize alpha/contribution from cells or
  faces with poor multiview witness; start diagnostic-only before training.

If the loss is a pure function of decoded tensors and CSR adjacency, keep it
in Python. If it uses per-ray interval endpoints, contribution mass, local
texel coordinates, or winning face ids, it belongs in aux/replay first and only
later in Metal backward.

## Forward/Backward Requirements For Common Directions

```text
Transport-only regularizer:
    forward support: decoded params + adjacency
    backward support: PyTorch autograd only
    checkpoint: no new required fields unless trainable transport params exist

New SV/color law:
    forward support: stream_sv_texel_color or new helper
    backward support: route color gradients to sv_axis/sv_rgb and any new axis
                      temperature/metric params
    test: direct fixture gradient keys plus Metal shared gradient parity

Height/normal law:
    forward support: stream_clip_height_surface and normal_distance aux
    backward support: grad_features for height and frame vectors; if normal is
                      no longer derivable from quaternions, add explicit tensor
    test: normal_distance loss backward smoke

Cell ownership/incidence law:
    forward support: interval clip helper and raytrace traversal
    backward support: dt/t_near/t_far derivatives to centers/radii and any new
                      incidence params
    test: tiled vs raytrace parity, backward finite-difference fixture

Fluid continuity law:
    forward support: usually none in rasterizer if only regularizing decoded
                     positions/material over time
    backward support: PyTorch autograd through centers/quaternions/texels
    test: train smoke plus checkpoint reload/drift metrics
```

## Config Surface

Keep config additions checked in under `src/train_configs/*.jsonc`.

Recommended sections:

```text
model:
    feature_mode
    adjacency_mode
    num_texel_sites
    sv_dof
    new trainable parameter shape/initialization only

render:
    use_tiled
    use_raytrace
    texel_temperature
    diagnostic depth quantiles or replay flags only if they affect rendering

losses:
    new scheduled weights
    robust penalty scale/delta
    witness thresholds only if used as loss

logging:
    extra artifact cadence if the diagnostic is expensive
```

Do not add environment-variable fanout for every knob. If an older config must
load, normalize the default once in `resolve_config`.

## Checkpoint And Artifact Contract

Current checkpoints save:

```text
model.state_dict()
serialized config
step
metrics
best metric name/value
```

New trainable tensors must be `nn.Parameter` or persistent buffers on
`MetalPowerFoamVideo` so they naturally land in `state_dict`. New nontrainable
diagnostic state should be a persistent buffer only if it is required for
resume semantics. Otherwise log it as artifact data.

Recommended artifacts for new math:

```text
eval_metrics_history.jsonl:
    heldout metrics + new scalar diagnostics

train_metrics_history.jsonl:
    loss component, scheduled weight, LR group if new params exist

best_metrics.json:
    best heldout selector plus diagnostic snapshot

checkpoint_best.pt:
    includes enough state to resume and re-render the selected behavior

diagnostic sidecars:
    face_witness_summary.json
    transport_residual_summary.json
    holonomy_summary.json
```

If an artifact claims 4K speed or trainability, route it through the existing
dynamic foam verifiers instead of relying on a screenshot or a single run log.

## Gradient And Metric Key Maps

Avoid expanding future tests and loggers into long dict destructuring blocks.
The existing fixture tests already use the right pattern:

```text
DIRECT_FIXTURE_GRAD_KEYS = (
    ("grad_points", "points"),
    ("grad_radii", "radii"),
    ...
)

METAL_SHARED_GRAD_KEYS = (
    ("grad_density", "densities"),
    ("grad_texel_height", "texel_height"),
    ("grad_texel_sv_axis", "texel_sv_axis"),
    ("grad_texel_sv_rgb", "texel_sv_rgb"),
)
```

Use the same style for new math:

```text
PARAM_SPECS = (
    ("centers", "points", (3,)),
    ("radii", "radii", ()),
    ("texel_sites", "texel_sites", (S,2)),
    ("texel_height", "texel_height", (S,)),
    ("sv_axis", "texel_sv_axis", (S,D,3)),
    ("sv_rgb", "texel_sv_rgb", (S,D,3)),
)

LOSS_SPECS = (
    ("connection", "connection_loss", "connection_weight"),
    ("transport", "transport_loss", "transport_weight"),
)

AUX_METRIC_KEYS = (
    "aux_mean_contrib",
    "aux_mean_point_error",
    "aux_mean_normal_distance",
    "aux_face_witness_p10",
    "aux_transport_residual_mean",
)
```

Rules:

- pass canonical containers (`cfg`, `model`, `metrics`, `params`) to helpers
  and let the leaf helper read the needed keys
- one key map should drive fixture extraction, allclose assertions, JSON
  serialization, and W&B payload naming where possible
- add a helper like `_assert_fixture_grads(expected, params, GRAD_KEYS, ...)`
  instead of adding another wall of local variables
- for metric payloads, map internal snake_case keys to W&B names in one loop
  rather than hand-writing every `payload["Group/Name"]`
- if a new tensor is packed into `features_flat`, keep exactly one packing map
  that states offset, shape, and gradient meaning

This is not style cleanup. It protects the behavioral contract: when a new
math channel gets a forward value, a gradient value, a metric, and a checkpoint
field, the same declared key map should make omissions visible.

## Minimal First Experiment Shape

The least risky implementation sequence for new gauge/fluid math is:

```text
1. Add diagnostic-only replay/aux metric.
2. Prove the diagnostic correlates with heldout residual, not source residual.
3. Add a Python regularizer over decoded tensors if possible.
4. Only then extend Metal feature packing or op schemas.
5. Run 1-step F=3/F32-style trainer smoke equivalent for this path.
6. Run PowerFoam direct/Metal focused tests and 4K verifier if making speed or
   trainability claims.
```

Acceptance should stay framed as heldout behavior and saved verifier artifacts.
A lower source-view loss from a new gauge/fluid term is not evidence that the
representation improved.
