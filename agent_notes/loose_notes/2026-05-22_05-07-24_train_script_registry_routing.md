# Train Script Registry Routing

## What Changed

Updated embedded Python snippets in scale/pretrain shell launchers so they use
the trainer registry for generic config resolution and dispatch:

- `src/train_scripts/train_scale_static_dynamic_vjepa_multicam.sh`
- `src/train_scripts/train_scale_static_dynamic_vjepa_1k_video_pretrain.sh`
- `src/train_scripts/train_single_video_pretrain_300_64f.sh`
- `src/train_scripts/train_single_video_pretrain_all_youtube_64f_512.sh`

Concrete trainer imports were replaced with:

- `trainer_registry.resolve_config_for_arch(...)` for config checks and prebake
  setup.
- `trainer_registry.run_config_dict(...)` for patched probe/run configs.

## Why This Slice

The shell launchers were importing
`PrecomputedFeatureImplicitTrainer.resolve_config(...)`,
`MulticamPrecomputedFeatureImplicitTrainer.resolve_config(...)`, or
`train_precomputed_feature_implicit_dynamic.run_training(...)` only because
they needed generic config resolution or in-memory dispatch. That made launch
scripts depend on concrete trainer class namespaces. The registry already owns
that boundary, so the scripts now match the rest of the cleanup direction.

## Validation

Concrete trainer-import scan:

```bash
rg -n "from train_precomputed_feature_implicit_dynamic import|from train_multicam_precomputed_feature_implicit_dynamic import|from train_video_token_implicit_dynamic import" src/train_scripts
```

Result: no matches.

Shell syntax:

```bash
bash -n \
  src/train_scripts/train_scale_static_dynamic_vjepa_multicam.sh \
  src/train_scripts/train_scale_static_dynamic_vjepa_1k_video_pretrain.sh \
  src/train_scripts/train_single_video_pretrain_300_64f.sh \
  src/train_scripts/train_single_video_pretrain_all_youtube_64f_512.sh
```

Result: passed.

Lightweight runtime checks:

```bash
src/train_scripts/train_single_video_pretrain_300_64f.sh resolve
src/train_scripts/train_scale_static_dynamic_vjepa_1k_video_pretrain.sh check
src/train_scripts/train_scale_static_dynamic_vjepa_multicam.sh check
```

Results:

- 300 resolve printed the resolved 512px overfit config payload with
  `approx_decoded_gaussians=8192`.
- 1k check resolved `arch=precomputed_feature_implicit_camera`, lazy manifest
  load mode, and manifest counts `{'train': 1000}`.
- Multicam check wrote a temporary config and resolved
  `arch=multicam_precomputed_feature_implicit_camera` for sample index 0.

## Next

Keep remaining direct trainer imports only when they are structural: subclass
inheritance, tests for trainer behavior, or diagnostics that instantiate a
specific model class. Do not use concrete trainers as generic config resolver or
runner namespaces.
