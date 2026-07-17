# STAR UVT Visibility-Proxy Trainer Gate

## Context

The previous gate proved the support-changing visibility idea only in a CPU toy
scene: same-support dense alpha could not move from zero target hits, while a
soft target-pixel to projected-tube coverage proxy sent center/velocity
gradients and reached `0.324` target alpha `>0.10` coverage.

This session ported that idea into `train_star_uvt_feature_overfit.py` so it can
run as a first-class trainer option instead of a side script.

## Implementation

Added a top-level `visibility_proxy` config block with validation in
`resolve_config`. The trainer now samples bright target pixels from the real
RGB target, computes a soft nearest projected-tube coverage loss from
`center_uv`, `center_t`, and `velocity_uv`, backprops it after the normal
feature/probe losses, and records timing, target point count, start/end losses,
and pass/fail metadata.

The gate config is:

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_visibility_proxy_from1500_lr001_5step_media.jsonc`

It resumes from:

`outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_1500step.pt`

## Run

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_visibility_proxy_from1500_lr001_5step_media.jsonc
```

Main artifacts:

- `outputs/benchmarks/2026-05-20_star_uvt_visibility_proxy_trainer_gate.md`
- `outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_visibility_proxy_from1500_lr001_5step_media.json`
- `outputs/checkpoints/2026-05-20_star_uvt_feature_targetgrid_visibility_proxy_from1500_lr001_5step.pt`
- `outputs/media/2026-05-20_star_uvt_feature_targetgrid_visibility_proxy_from1500_lr001_5step_contact.jpg`
- `outputs/media/2026-05-20_star_uvt_feature_targetgrid_visibility_proxy_from1500_lr001_5step_probe_contact.jpg`

## Result

The first-class trainer mechanics gate passes:

- weighted loss `0.8719856 -> 0.8718643`
- feature target loss `0.6254179 -> 0.6253786`
- RGB-probe loss `0.0062694 -> 0.0062674`
- RGB-probe PSNR `22.0277 -> 22.0291`
- visibility-proxy loss `-4.2095666 -> -4.2099228`
- `visibility_proxy_target_point_count=4096`
- center, velocity, feature, opacity, and precision gradients are seen
- final dense full RGB PSNR is still only `5.640`

Mean timing over five steps is `541.07ms` step, `306.65ms` backward,
`141.45ms` render forward, and `236.98ms` visibility proxy. Last step is
`397.49ms` step, `286.14ms` backward, and `211.59ms` visibility proxy.

## Interpretation

This is real progress on the current goal because the support-changing lever is
no longer just a CPU toy. It is now in the actual STAR UVT trainer with config
validation, checkpoint/report plumbing, gradient checks, and a passing real
video 512px/64f gate.

It is not a visual-quality promotion. Dense RGB is still bad, and the proxy is
expensive enough that scaling it blindly would be wasteful. The right next work
is either a longer continuation that measures whether this proxy actually
changes dense support/alpha coverage, or a fused/cheaper visibility proxy if the
overhead remains near 200-300 ms per step.
