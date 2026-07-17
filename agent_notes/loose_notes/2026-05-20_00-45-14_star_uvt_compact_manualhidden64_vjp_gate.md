# STAR UVT Compact Manual-Hidden64 VJP Gate

Date: 2026-05-20 00:45 +07

## Goal

Close the obvious gap after the compact native star-only rejection: test whether
the compact target-area visual route can keep colorizer gradients while avoiding
PyTorch autograd overhead for the hidden64 `FeatureToColor` VJP.

## Added

- Config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_manualhidden64_from1500_lr001_5step_diagnostic.jsonc`
- Report generator:
  `research_experiments/star_uvt_feature_tubes/compare_compact_visual_vjp_gate.py`
- Generated comparison report:
  `outputs/benchmarks/2026-05-20_star_uvt_compact_visual_vjp_gate.md`

## Command

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_manualhidden64_from1500_lr001_5step_diagnostic.jsonc

rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/compare_compact_visual_vjp_gate.py
```

W&B offline run: `jq218re1`

Output JSON:
`outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_compact_manualhidden64_from1500_lr001_5step_diagnostic.json`

## Result

Rejected.

The diagnostic proves colorizer gradients are present:

- `colorizer_grad_required=true`
- `colorizer_grad_seen=true`

But it is slower and quality-negative:

- mean/no-first step: `2007.38/1899.16ms`
- mean/no-first backward: `952.49/859.60ms`
- mean sparse visual loss/backward: `415.66/341.19ms`
- weighted loss: `1.146733 -> 1.153926`
- feature loss: `0.625418 -> 0.626795`
- RGB-probe PSNR: `22.0277 -> 21.8596`
- sparse visual loss: `0.270538 -> 0.266454`
- zero tile overflow

The comparison report uses the first five steps from the compact-autograd
50-step keeper for a matched timing window:

- compact autograd: `991.87ms` mean step, `787.75ms` no-first step,
  `620.75ms` mean backward, `507.56ms` no-first backward
- compact native star-only vec4 W^T: `2265.02ms` mean step, no colorizer grads
- compact manual hidden64: `2007.38ms` mean step, colorizer grads present but
  feature/probe quality regresses

## Decision

Keep `star-feature-512-visual` / compact autograd as the practical single-video
visual overfit route.

Do not port the manual hidden64 compact path as-is. It answers the gradient
contract but misses the speed and quality gate. The next native attempt only
matters if it returns colorizer parameter gradients and beats the compact
autograd loss+backward envelope, not merely the rejected full-cell8 or star-only
native paths.
