# STAR UVT Precomputed V-JEPA Bridge Audit

Date: 2026-05-19 03:37 ICT

## Question

Do we already have multi-resolution STAR UVT running with precomputed V-JEPA
features in the fastest UVT STAR feature route?

## Answer

No. The selected fast route is still RGB reconstruction through
`FeatureToColor`, not cached V-JEPA target training.

Checked fast route:

```text
src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-feature-512-fast
src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_20step_media.jsonc
```

That config is `arch=star_uvt_feature_overfit`, uses
`feature_direct_gradcache_reduce_vec4`, has `colorize.pre_norm=false`, renders
64 frames at 512px with 8192 F32 tubes, and has no `features` section.

## Evidence

Audit command:

```bash
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_bridge_audit.py \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.json \
  --out-md outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md
```

Audit result:

```text
fastest_star_uvt_feature_route_uses_precomputed_vjepa: false
precomputed_vjepa_exists_elsewhere: true
```

Code contract from the audit:

```text
dispatcher_has_star_uvt_feature_route: true
star_feature_trainer_uses_rgb_video_target: true
star_feature_trainer_uses_video_feature_cache: false
```

Initial config inventory before the `rgb_pyramid` smoke config was added:

```text
STAR UVT feature configs scanned: 43
STAR UVT feature configs with V-JEPA/precomputed sections: 0
precomputed V-JEPA Gaussian/token configs found: 35
```

After the cached-target smoke implementation, rerunning the audit reports:

```text
STAR UVT feature configs scanned: 44
STAR UVT feature configs with V-JEPA/precomputed sections: 0
STAR UVT feature configs with cached-target smoke enabled: 1
precomputed V-JEPA Gaussian/token configs found: 35
STAR feature trainer uses VideoFeatureCache: true
STAR feature trainer has cached-target adapter: true
```

## Interpretation

The fast shader work selected a usable speed diagnostic for STAR UVT feature
tubes, but it did not solve the target representation. The route renders dense
F32 tube features, decodes them with `FeatureToColor`, and optimizes RGB video
reconstruction.

Cached V-JEPA already exists in the repo through the
`precomputed_feature_implicit_camera` /
`multicam_precomputed_feature_implicit_camera` Gaussian-token trainer family.
The STAR feature trainer now has the generic cached-target adapter, but no STAR
UVT feature config uses real V-JEPA targets yet.

Later in the same continuation, the 8f/64px real-V-JEPA smoke config was added
and passed. The selected 512px fast diagnostic still has no `features` section,
but STAR now has a separate real cached-V-JEPA target smoke row:

```text
src/train_configs/star_uvt_feature_testvideo_8f_64_vjepa_target_gradcache_reduce_vec4_10step.jsonc
loss 1.0008157045 -> 0.9004198760
token grid [4,16,16], source [1,1024,768], adapted [8,32,64,64]
```

## Next Bridge Contract

1. Keep the `rgb_pyramid` cache smoke as the cheap bridge regression.
2. Keep the real V-JEPA smoke as the cached-feature regression.
3. Scale the V-JEPA target contract to the selected no-pre-norm
   `feature_direct_gradcache_reduce_vec4` 512px renderer.
4. Only then scale to the prepared 300-video set and compare against
   Gaussian/token V-JEPA baselines with explicit `BASELINES.md` rows.

## Updated Docs

- `README.md`
- `PROJECT_INDEX.md`
- `EXPERIMENTS.md`
- `TODO/README.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `agent_notes/key_learnings.md`

## Validation

Validation to run after doc edits:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_bridge_audit.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_precomputed_feature_implicit_dynamic.py
wc -l agent_notes/key_learnings.md
git diff --check
git -C third_party/fast-mac-gsplat diff --check
pgrep -fl 'train.py|train_star_uvt_feature_overfit|star_uvt_vjepa_bridge_audit'
```
