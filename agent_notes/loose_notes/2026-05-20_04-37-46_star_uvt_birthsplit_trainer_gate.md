# STAR UVT Birth/Split Trainer Gate

## Context

The earlier support work rejected same-support pressure:

- Center-only visibility proxy passed mechanics but did not move dense support enough.
- Opacity/precision support proxy sent the missing gradients but was slow and barely changed support.
- CPU birth/split was the first positive mechanism: reallocating dead tubes onto target support produced coverage from a zero-hit fixture.

The next missing step was to port that fixed-budget support change into the real STAR UVT feature trainer.

## Implementation

Added `support_birth_split` to `src/train/train_star_uvt_feature_overfit.py`.

The opt-in path:

- samples target points from high-brightness target RGB locations using the existing target-point sampler;
- fits a linear screen-space trajectory in centered STAR time;
- selects a fixed number of existing tubes by `tube_selection=lowest_opacity` or `first`;
- reallocates those tubes in place, preserving the total tube budget;
- sets center, velocity, broad spatial/temporal precision, and opacity before the train loop;
- records the full birth/split state in the output JSON.

The config validation rejects checkpoint resume with optimizer state when birth/split is enabled, because reallocated parameters should not inherit stale Adam moments.

New config:

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_from1500_lr001_5step_media.jsonc`

## Validation

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python -m py_compile \
  src/train/train_star_uvt_feature_overfit.py \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_visibility_support_bridge.py \
  research_experiments/star_uvt_feature_tubes/visibility_support_birth_split_prototype.py
```

Passed.

```bash
PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_visibility_support_bridge.py -q
```

`34 passed in 1.68s`.

## Trainer Gate

```bash
PYTHONPATH=src/train rtk uv run python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_from1500_lr001_5step_media.jsonc
```

Result JSON:

`outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_birthsplit32_from1500_lr001_5step_media.json`

Report:

`outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_trainer_gate.md`

W&B: offline run `zinvh2gf`.

The 5-step gate passed:

- weighted loss `0.910290 -> 0.909536`;
- feature target loss `0.635579 -> 0.635530`;
- frozen RGB-probe loss `0.006868 -> 0.006850`;
- full RGB PSNR `5.708`;
- mean step/backward/render `189.4/55.6/70.1ms`;
- last step `138.3ms`;
- zero overflow, max/p95/cap `100/71/128`.

Birth/split state:

- target samples `2048`;
- reallocated tubes `32 / 8192`;
- selected opacity mean `0.3418 -> 0.8000`;
- fitted center `[218.973, 326.002]`;
- fitted velocity `[-1.157, 1.120]`;
- spatial/temporal precision `0.00061035`;
- tube budget preserved.

## Interpretation

This closes the trainer-port gap: fixed-budget support birth/split is no longer just a CPU mechanism. It is cheaper than the all-tube support proxy and keeps zero overflow.

It does not complete the visual-quality goal. The full RGB PSNR is slightly higher than center/support proxy rows (`5.708` vs `5.640`/`5.643`) but still not close to a usable overfit, and the run starts with deliberately changed support from a selected sparse checkpoint. Treat this as a viable support-changing primitive.

## Dense Support Diagnostic

Follow-up:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  --case start1500=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume50_from1450_lr005sparse_media.jsonc \
  --case center5=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_visibility_proxy_from1500_lr001_5step_media.jsonc \
  --case support5=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_visibility_support_from1500_lr001_5step_media.jsonc \
  --case birth32=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_from1500_lr001_5step_media.jsonc \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_dense_support_diagnostic.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_dense_support_diagnostic.md \
  --date 2026-05-20
```

Result:

- `start1500`: normal `5.438`, forced-alpha `11.722`, oracle `20.140`, alpha `>0.1` `0.411`.
- `center5`: normal `5.640`, forced-alpha `14.552`, oracle `25.834`, alpha `>0.1` `0.405`.
- `support5`: normal `5.643`, forced-alpha `14.553`, oracle `25.820`, alpha `>0.1` `0.406`.
- `birth32`: normal `5.708`, forced-alpha `14.606`, oracle `25.234`, alpha `>0.1` `0.411`, alpha `>0.5` `0.117`.

Read: birth/split improves black-background PSNR, forced-alpha PSNR, alpha mean, and high-alpha support versus center/support proxy rows. It does not solve coverage, because alpha `>0.1` only returns to the start1500 level and target-background oracle falls versus center/support.

## Next

Sweep `reallocate_tubes` and radius before any 50-step or 300-video run. Also change target sampling from top-brightness points to uncovered/low-alpha target pixels; the dense diagnostic suggests the current target samples add some support but not enough missing coverage.
