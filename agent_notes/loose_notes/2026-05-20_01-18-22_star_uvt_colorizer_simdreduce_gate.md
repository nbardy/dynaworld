# STAR UVT Colorizer SIMD-Reduce Gate

Date: 2026-05-20 01:18 +07

## Goal

After the native colorizer atomic split and Torch sidecar reducer diagnostics,
test the smallest same-pass native colorizer-gradient reduction. The hypothesis
was that SIMD-group pre-reduction before global colorizer parameter atomics
could keep the compact native target-area route correct while removing the
dominant atomic pressure.

## Implementation

Changed the STAR UVT native target-area backward path in
`third_party/fast-mac-gsplat/variants/star_uvt_v0`:

- added `simd_reduced_atomic_add(...)`;
- threaded `thread_index_in_simdgroup` into the target-area backward kernel;
- used `mode_bits & 256u` to enable SIMD-reduced colorizer parameter atomics;
- added Python modes `target_area_colorizer_simdreduce_grad_only` and
  `target_area_colorizer_simdreduce_vec4_wt`;
- added trainer loss modes
  `native_hidden64_target_area_colorizer_simdreduce_vec4_wt` and
  `native_hidden_target_area_colorizer_simdreduce_vec4_wt`;
- added the 5-step diagnostic config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_nativecolorizer_simdreduce_vec4wt_from1500_lr001_5step_diagnostic.jsonc`.

Native rebuild command:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0
  rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

The build passed.

## Direct Kernel Results

Tiny parity passed for both new modes:

- `outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_simdreduce_vec4_wt_tiny_gate.json`
  - pass true
  - native total/backward `14.36/7.70ms`
  - F32 max errors: feature `2.33e-10`, hidden weight `1.16e-10`, hidden bias
    `2.33e-10`, output weight `5.82e-10`, output bias `1.86e-09`, loss `0`
- `outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_simdreduce_grad_only_tiny_gate.json`
  - pass true
  - native total/backward `10.53/6.20ms`
  - F32 colorizer-gradient errors <= `4.66e-10`, loss `0`

Compact support, `64f/512px/8192t`, F32, `grid_side=64`, `patch_size=2`,
`tile_capacity=128`, `6.25%` dense support:

- prior star-only vec4 W^T: `146.62ms` total, `88.89ms` backward
- prior naive colorizer-grad-only: `727.32ms` total, `536.57ms` backward
- prior naive full colorizer vec4 W^T: `571.21ms` total, `531.40ms` backward
- SIMD-reduce colorizer-grad-only: `315.02ms` total, `248.12ms` backward
- SIMD-reduce full colorizer vec4 W^T: `330.36ms` total, `249.27ms` backward
- SIMD-reduce full matched baseline run: native `297.20ms` total,
  `239.23ms` backward; sparse-pixel baseline `312.07ms` total
  (`9.63/270.85/31.59ms` render/loss/backward)

Direct-kernel decision: same-pass SIMD reduction is a real fix for the naive
colorizer atomic envelope.

## Trainer Result

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_nativecolorizer_simdreduce_vec4wt_from1500_lr001_5step_diagnostic.jsonc
```

Output JSON:
`outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_sparsevisual_targetarea64_compact_nativecolorizer_simdreduce_vec4wt_from1500_lr001_5step_diagnostic.json`

Offline W&B run: `8dhyt3i1`.

Result:

- pass false
- colorizer grad required/seen true/true
- zero tile overflow, max/p95/cap `68/46/128`
- mean step/backward `2908.91/1362.97ms`
- last step/backward `2979.20/1150.71ms`
- sparse visual render/loss/backward mean `256.08/456.35/604.00ms`
- feature target loss worsens `0.625418 -> 0.626795`
- RGB probe PSNR drops `22.0277 -> 21.8596`
- sparse visual loss improves `0.270538 -> 0.266454`

Against the old native colorizer vec4 W^T diagnostic, SIMD-reduce cuts the
native sparse-visual backward slice (`871.46 -> 604.00ms`) and mean backward
(`1474.24 -> 1362.97ms`), but the full step is still worse than compact
autograd and the same quality regression remains.

## Decision

Do not promote the SIMD-reduced native colorizer route to the trainer helper.
It is a good direct-kernel primitive, but the single-video visual route remains:

```bash
WANDB_MODE=offline rtk ./src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-feature-512-visual
```

What is left is not "prove colorizer atomics can be reduced"; that is done. The
remaining work is either a native path that also removes duplicated target-area
forward/loss overhead and preserves quality, or a different visual objective
that actually improves dense RGB/contact-sheet quality.

Report:
`outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_simdreduce_gate.md`
