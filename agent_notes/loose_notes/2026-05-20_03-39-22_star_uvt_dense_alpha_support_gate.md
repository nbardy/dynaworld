# STAR UVT Dense Alpha Support Gate

Date: 2026-05-20

## Context

The current STAR UVT feature-tube route is speed-usable through sparse-forward
batched target-grid/probe VJP, but visual quality is blocked by sparse/weak
dense support. Recent gates rejected RGB-grid, sampled alpha-to-one, phase
coverage, black-hole target energy, target-background composition as a route,
patch4 support, and raw opacity bias. The remaining narrow question was whether
dense alpha supervision could move support without the hidden64 visual VJP cost.

## Change

Added an opt-in `dense_alpha` section to
`src/train/train_star_uvt_feature_overfit.py`:

- `dense_alpha.enabled`
- `dense_alpha.loss_weight`
- `dense_alpha.alpha_target`
- `dense_alpha.backward_mode`

The trainer now logs dense-alpha loss series and render/loss/backward timings,
saves the series in checkpoints, and includes dense-alpha loss decrease in the
pass gate when enabled. The default backward mode is
`gradcache_skip_feature_grad`, so this path sends dense `grad_alpha` through the
STAR UVT renderer while skipping feature-gradient atomics.

New pilot config:

`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_densealpha075_from1500_lr001_5step_media.jsonc`

This starts from the selected sparse 1500 checkpoint and keeps feature/probe on
the selected sparse-forward batched target-grid route. Sparse visual loss is
disabled to isolate dense alpha.

## Commands

Validation before launch:

```bash
rtk .venv/bin/python -m py_compile src/train/train_star_uvt_feature_overfit.py tests/test_star_uvt_feature_target_adapter.py research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py
PYTHONPATH=src/train rtk uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py tests/test_star_uvt_feature_rgb_probe.py -q
PYTHONPATH=src/train rtk .venv/bin/python - <<'PY'
from pathlib import Path
from config_utils import load_config_file
from train_star_uvt_feature_overfit import resolve_config
p=Path('src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_densealpha075_from1500_lr001_5step_media.jsonc')
cfg=resolve_config(load_config_file(p))
print(cfg['dense_alpha'])
print(cfg['output']['out_json'])
PY
```

Benchmark:

```bash
PYTHONPATH=src/train rtk .venv/bin/python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_densealpha075_from1500_lr001_5step_media.jsonc
```

Dense diagnostic:

```bash
PYTHONPATH=src/train rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  --case compact=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_media.jsonc \
  --case densealpha075=src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_densealpha075_from1500_lr001_5step_media.jsonc \
  --raw-opacity-biases=-2,-1,0,1,2,3,4 \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_densealpha075_dense_diagnostic.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_densealpha075_dense_diagnostic.md
```

## Results

Focused tests passed: `38 passed`.

The trainer wrote artifacts and then exited with code `1` because
`require_loss_decrease` correctly rejected the endpoint:

- Result JSON:
  `outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_densealpha075_from1500_lr001_5step_media.json`
- Checkpoint:
  `outputs/checkpoints/2026-05-20_star_uvt_feature_targetgrid_densealpha075_from1500_lr001_5step.pt`
- W&B offline run:
  `wandb/offline-run-20260520_031122-ikch7juj`
- Weighted loss: `1.271702 -> 1.284505`
- Dense alpha loss: `0.395507 -> 0.397107`
- Feature loss: `0.625418 -> 0.626814`
- RGB-probe PSNR: `22.028 -> 21.861`
- Dense full RGB PSNR: `5.647`
- Mean step/backward: `2558.64/1114.22ms`
- Dense-alpha render/loss/backward: `834.45/124.58/858.91ms`
- Tile overflow: `0`

Dense diagnostic:

| case | normal PSNR | forced-alpha PSNR | target-bg oracle PSNR | best gain PSNR | best floor PSNR | best raw-opacity PSNR | alpha > 0.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compact | 6.023 | 11.450 | 20.149 | 7.861 @ 16x | 14.203 @ 0.75 | 6.194 @ +4 | 43.5% |
| densealpha075 | 5.647 | 14.556 | 25.809 | 8.279 @ 16x | 14.556 @ 1.0 | 5.816 @ +4 | 40.7% |

Report:

`outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_support_gate.md`

## Read

Dense alpha-only support is negative in this form. It is cheaper than full
hidden64 dense visual VJP but still expensive at 512px, and it does not create
the missing dense support. The first few steps reduce dense alpha loss, but the
5-step endpoint spikes and degrades feature/probe. The final dense diagnostic
shows stronger forced-alpha/oracle color potential but lower actual alpha
coverage and worse black-background composite quality.

This narrows the next branch: the missing piece is not another scalar alpha
pressure term. It is a support/visibility representation problem. The next
useful STAR UVT experiment should be a compact visibility/prefix tape or a
changed support parameterization/objective, with dense alpha treated as a
diagnostic gradient source only if it can be fused into that support-changing
mechanism.
