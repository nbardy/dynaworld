# STAR UVT Binner Binfix And Train Gate

Date:
    2026-05-26

## Context

The selected support-target patch diagnostic was intended to decide whether the
STAR support-birth plateau was a local target-patch issue or a broader
visibility/composition issue. It instead exposed a lower-level renderer bug:
selected born tubes had high analytic alpha at selected target points, but
`selected_only` sparse rendering returned zero alpha and zero tile counts.

## Current Model

STAR UVT local frame chunks shift tube centers in time. A tube born with
global `m.z ~= 0` can enter a 2-frame chunk around global frame 44 with local
`m.z ~= -13`. For a moving UVT tube, the valid time support comes from the
inverse 3x3 precision diagonal. The determinant scale for wide temporal tubes
is small (`~1e-9`), so using `eps=1e-8` as a determinant cutoff falsely
rejected valid inverse bounds. The old fallback then used temporal half-extent
`mi.frames`, which is not enough to include a shifted center outside the local
chunk.

## Changes

- Patched `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`.
- `inverse_sym3_diag` now accepts determinants above `max(eps^2, 1e-20)`.
- `tube_bounds` fallback now uses `abs(m) + domain` so fallback bounds cover
  shifted local chunks instead of only the immediate frame count.
- Added `tests/test_star_uvt_feature_binning.py`, a focused MPS regression for
  a moving tube that should render inside a chunk even when its shifted center
  is outside that chunk.
- Added a fresh binfix train config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetarea2_binfix_from1500_lr001_50step_media.jsonc`.

## Evidence

Focused test:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_feature_binning.py -q
1 passed
```

Repaired three-case selected-patch diagnostic:

```text
outputs/benchmarks/2026-05-26_star_uvt_support_target_patch_diagnostic_binfix.md

targetinit  normal patch PSNR 4.606, forced 14.529, oracle 26.103, alpha 0.337
targetalpha normal patch PSNR 4.686, forced 14.694, oracle 26.269, alpha 0.344
targetarea2 normal patch PSNR 4.684, forced 14.677, oracle 26.202, alpha 0.344
selected_only alpha ~= 0.30 across cases
```

Before the fix, the same diagnostic had `selected_only` alpha `0.0` despite
analytic selected-tube alpha mean/max around `0.20/0.36`.

First repaired 50-step train:

```text
outputs/benchmarks/2026-05-26_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_binfix_50step_media.json

loss:                    0.889263 -> 0.863064
feature loss:            0.612217 -> 0.610967
RGB-probe PSNR:          24.253 -> 24.453
support-target-area loss: 0.253626 -> 0.217254
tile overflow:           0
max tile count:          110 / 128
```

Post-train selected-patch diagnostic:

```text
outputs/benchmarks/2026-05-26_star_uvt_support_target_patch_diagnostic_targetarea2_binfix_train.md

normal/forced/oracle patch PSNR: 6.644 / 19.452 / 26.994
patch alpha mean:                0.481
selected-only alpha:             0.444
```

Post-train dense support diagnostic:

```text
outputs/benchmarks/2026-05-26_star_uvt_birthsplit_targetarea2_binfix_dense_support.md

normal/forced/oracle dense PSNR: 7.269 / 14.736 / 21.439
alpha mean:                     0.3456
alpha > 0.1:                    75.4%
alpha > 0.5:                    29.1%
best posthoc alpha gain:        12.399 PSNR @ 16x
best alpha floor:               14.736 PSNR @ 1.0
best raw-opacity bias:          8.039 PSNR @ +4
```

Compared with the pre-binfix targetarea2 repair dense row
(`6.507/14.085/21.627`, alpha `>0.1` `65.7%`), the shader repair did improve
whole-frame support and forced-alpha content. It did not make opacity bias or
local target-area pressure sufficient: the target-background oracle remains
around `21.4` PSNR, and raw opacity bias barely reaches `8.0` PSNR.

Visibility-prefix tape diagnostic:

```text
outputs/benchmarks/2026-05-26_star_uvt_targetarea2_binfix_visibility_prefix_tape.md

sampled selected support-target rays: 256
normal/forced/oracle sampled PSNR:    6.522 / 19.129 / 26.831
final alpha mean:                     0.4755
selected alpha max mean:              0.2670
selected weight sum mean:             0.4363
selected weight share mean:           0.9308
selected prefix at alpha max mean:    0.8734
selected absent fraction:             0.0%
selected prefix-hidden fraction:      1.6%
top contributor selected fraction:    95.7%
```

This is not an "old tubes hide born tubes" result. On selected target rays,
the born tubes usually own the prefix tape already. The sampled-ray failure is
that ownership does not become enough final alpha over black background, while
the whole-frame dense failure says this ownership also needs to spread beyond
the selected rays.

## Decision Implication

The selected born tubes now survive sparse binning and carry real local target
patch mass. Dense transfer is partial, not decisive: enough to prove the binner
fix matters, not enough to promote the current support-target-area objective.
The prefix tape now says selected support is present and dominant locally, so
the next train should target prefix-weight/final-alpha composition and broadened
coverage sampling. Softmax-GS and WorldFoam remain parked until STAR's repaired
support reaches a clean visibility/composition decision or a matched
representation tournament gives them heldout evidence.

## 2026-05-28 Prefix-Alpha Follow-Up

Implemented and tested a compact autograd prefix-alpha loss in
`src/train/star_uvt_feature_overfit_trainer.py`, with config keys in
`src/train/star_uvt_feature_config.py`:

```text
support_birth_split.prefix_alpha_loss_weight
support_birth_split.prefix_alpha_target
support_birth_split.prefix_alpha_max_points
```

The loss sorts contributors by depth at selected support-birth target points,
computes alpha-over prefix weights, sums the selected born-tube contribution,
and pushes that selected contribution toward a target final alpha. It treats
depth order as a fixed/no-grad ordering surface and lets gradients flow through
the selected alpha/opacity/shape terms.

Focused tests:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_binning.py \
  tests/test_star_uvt_visibility_prefix_tape_diagnostic.py -q

2 passed
```

Short 20-step probe:

```text
config:
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetarea2_binfix_prefixalpha_from1500_lr001_20step_media.jsonc

row:
outputs/benchmarks/2026-05-28_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_binfix_prefixalpha085w2_20step_media.json

pass: true
loss: 1.285825 -> 1.253901
support-target-area loss: 0.253626 -> 0.239840
prefix-alpha loss: 0.198281 -> 0.188115
selected weight: 0.4114 -> 0.4234
final alpha: 0.4456 -> 0.4571
tile overflow: 0
```

Fair 50-step comparison against the no-prefix binfix row:

```text
config:
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetarea2_binfix_prefixalpha_from1500_lr001_50step_media.jsonc

row:
outputs/benchmarks/2026-05-28_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_binfix_prefixalpha085w2_50step_media.json

pass: true
loss: 1.285825 -> 1.210325
support-target-area loss: 0.253626 -> 0.219786
prefix-alpha loss: 0.198281 -> 0.172906
selected weight: 0.4114 -> 0.4419
selected share: 0.9333 -> 0.9382
final alpha: 0.4456 -> 0.4751
tile overflow: 0
```

Dense support diagnostic for the 50-step prefix-alpha row:

```text
outputs/benchmarks/2026-05-28_star_uvt_birthsplit_targetarea2_binfix_prefixalpha085w2_50step_dense_support.md

normal/forced/oracle dense PSNR: 7.262 / 14.732 / 21.438
alpha mean:                     0.3452
alpha > 0.1:                    75.4%
best posthoc alpha gain:        12.394 PSNR @ 16x
best alpha floor:               14.732 PSNR @ 1.0
best raw-opacity bias:          8.037 PSNR @ +4
```

Visibility-prefix tape diagnostic for the same checkpoint:

```text
outputs/benchmarks/2026-05-28_star_uvt_targetarea2_binfix_prefixalpha085w2_50step_visibility_prefix_tape.md

sampled selected support-target rays: 256
normal/forced/oracle sampled PSNR:    6.465 / 19.051 / 26.734
final alpha mean:                     0.4719
selected weight sum mean:             0.4374
selected weight share mean:           0.9381
selected absent fraction:             0.0%
selected prefix-hidden fraction:      0.8%
top contributor selected fraction:    96.9%
```

Read: prefix-alpha is useful as a measurement/control surface and it does move
selected contribution on the rays it supervises. It does not beat the no-prefix
binfix dense row (`7.269/14.736/21.439`) at the same 50-step budget. The next
STAR support move should broaden target ownership/coverage or change the
sampling/support distribution; repeating local alpha pressure without changing
which tubes/points own dense pixels is unlikely to close the gap.
