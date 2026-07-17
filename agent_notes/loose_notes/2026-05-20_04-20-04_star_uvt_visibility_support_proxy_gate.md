# STAR UVT Visibility Support Proxy Gate

Date: 2026-05-20

## Original Goal Context

The active goal was to repeat and harden the STAR UVT fast feature-shader plan,
fill missing implementation details, execute the plan gate by gate, benchmark
each step, and record progress in markdown. The immediate open question after
the center-only visibility-proxy trainer gate was whether we had actually
changed dense support/alpha coverage, not just passed a proxy loss.

## What Changed

The center-only `visibility_proxy` in
`src/train/train_star_uvt_feature_overfit.py` was extended with
`center_weight`, `support_weight`, and `support_epsilon`.

The new support term evaluates projected tube coverage at sampled target points
using opacity and precision. That makes the proxy send gradients through
`raw_opacity` and `raw_precision`, not only center/velocity.

Focused tests were added to `tests/test_star_uvt_feature_target_adapter.py`:

- config resolution covers the new keys
- both proxy weights set to zero is rejected
- support-only proxy backprop produces nonzero opacity and precision gradients

## Trainer Gate

Config:

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_visibility_support_from1500_lr001_5step_media.jsonc`

Result:

- pass: true
- weighted loss: `0.910498 -> 0.909964`
- feature target loss: `0.625418 -> 0.625436`
- RGB-probe loss: `0.006269 -> 0.006268`
- RGB-probe PSNR: `22.0277 -> 22.0289`
- support proxy loss: `3.4303 -> 3.3821`
- mean step/backward/proxy: `1186.8/841.8/693.7ms`
- gradients seen: center, velocity, raw feature, raw opacity, raw precision

Artifacts:

- `outputs/benchmarks/2026-05-20_star_uvt_visibility_support_proxy_gate.md`
- `outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_visibility_support_from1500_lr001_5step_media.json`
- `outputs/run_logs/star_uvt_visibility_support_from1500_lr001_5step_20260520_041725.log`
- `outputs/checkpoints/2026-05-20_star_uvt_feature_targetgrid_visibility_support_from1500_lr001_5step.pt`

## Dense Support Diagnostic

Diagnostic artifact:

`outputs/benchmarks/2026-05-20_star_uvt_visibility_support_dense_diagnostic.md`

The diagnostic compared the selected sparse step-1500 checkpoint, the earlier
center-only 5-step proxy, and the new support-aware 5-step proxy.

| Checkpoint | Normal PSNR | Forced-alpha PSNR | Target-bg oracle | Alpha >0.1 |
| --- | ---: | ---: | ---: | ---: |
| start1500 | `5.438` | `11.722` | `20.140` | `0.411` |
| center5 | `5.640` | `14.552` | `25.834` | `0.405` |
| support5 | `5.643` | `14.553` | `25.820` | `0.406` |

## Read

This was real implementation progress but not a model-quality breakthrough.
The support-aware proxy closes the missing opacity/precision-gradient plumbing,
but it barely changes dense support and is far too expensive as written.

Do not scale this run to the 300-video set. The next experiment should either
fuse/cheapen the support objective or change the representation/objective so
dense coverage actually moves. Good candidates are explicit support-density
targets, support birth/split, or a tile-local support objective that avoids
the current sampled-target all-tube coverage cost.

## Validation

- `py_compile` passed for the trainer and focused test file before the run.
- Focused pytest for the adapter tests passed before the run.
- Trainer gate exited successfully.
- Dense support diagnostic exited successfully.

Final doc-sync validation is still required after updating project routing
docs.
