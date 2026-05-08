# Relpose Nonzero Output Init

## Context

The alpha `1/128` F32 multicam checkpoint had good heldout metrics but measured
near-identity predicted camera offsets. User feedback: pinhole is acceptable for
now if intrinsics/extrinsics are wide enough, but the relpose head should not
start every query camera at exactly the same source pose. This mirrors the
earlier Gaussian-head init work where nonzero head output weights and broad
token/point initialization improved coverage.

## Change

Added `train.relpose_output_init_std`.

- default: `0.0`, preserving old zero-init behavior
- when positive: initialize the final 6D relpose output layer weights with
  `Normal(0, relpose_output_init_std)` and zero bias
- outputs are still bounded by the existing `tanh` caps:
  `relpose_max_rotation_degrees` and `relpose_max_translation_ratio`

Why not replace this with a full look-at-orbit camera prior immediately:

- the current full relpose head emits an absolute source-relative SE(3) from
  source/query feature memories, not an explicit orbit-camera parameterization
- the base source camera and calibrated target templates remain the scene
  reference; the new init gives the head feature-dependent camera spread while
  keeping the old caps as guardrails
- a stronger orbit/look-at prior should be a separate architecture/config A/B,
  not silently mixed into this wiring change

## New Config

Created:

```text
src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_outputinit012.jsonc
```

It preserves the promoted alpha-threshold setup:

- train `camera_0006`, `camera_0014`
- heldout `camera_0005`
- `alpha_threshold = 1/128`
- F32 feature splatting
- 256px, 16 frames, 8192 splats
- `feature_variant = "v5_features"`

and changes:

```json
"relpose_output_init_std": 0.12
```

with a separate checkpoint path and W&B name:

```text
multicam-deepview-3cam-goodset-train0006-0014-holdout0005-vjepa-full-relpose-features-F32-256px-alpha1-128-relpose-outputinit012
```

## Expected Read

This should make initial relpose predictions nonzero and memory-dependent
instead of all identity/source-pose. It does not guarantee the cameras are
physically correct or perfectly look-at-origin; the next run still needs the
pose-error diagnostics requested in the previous visual-followup note.

## Initial Probe

Loaded the new config with `WANDB_MODE=disabled`, no checkpoint, and cached
V-JEPA features. Before any optimizer step, the relpose head predicted:

| Pair | Initial predicted offset | Calibrated/oracle offset | Initial error |
| --- | ---: | ---: | ---: |
| `camera_0006 -> camera_0006` | `17.544 deg`, `1.4715` translation | `0.000 deg`, `0.0000` | `17.544 deg`, `1.4715` |
| `camera_0006 -> camera_0014` | `16.417 deg`, `1.5571` translation | `33.186 deg`, `0.3094` | `24.609 deg`, `1.7267` |
| `camera_0006 -> camera_0005` | `16.200 deg`, `1.5509` translation | `18.631 deg`, `0.1851` | `13.983 deg`, `1.6933` |

This is a deliberately strong spread, not a pose-quality claim. The self-pair
starts wrong and must be pulled back by self reconstruction plus the identity
term; the useful part is that cross/heldout queries no longer begin collapsed
to source identity.
