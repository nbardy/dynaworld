# RGB Uniform Init Ablation Result

## Context

The user ran:

```text
PYTHONPATH=src/train uv run python src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_ablate_init_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc
```

Run:

```text
https://wandb.ai/nbardy/dynaworld/runs/d8i7kw6f
```

This config keeps the local non-pretrained video encoder, implicit camera,
16-frame clips, 128px input/render size, 128 tokens x 64 splats/token = 8192
Gaussians, and fast-mac renderer. The intended change is only initialization:

- `rgb_init="uniform"`, `rgb_init_min=0.0`, `rgb_init_max=1.0`
- `query_token_init_std=0.8`
- `head_output_init_std=0.12`

This run used the old trainer code before step-0 logging was added, so it has
no initialization media at W&B step 0.

Additional caveat found after the run: the local fast-mac wrapper had an unsafe
default `inputs_sorted_by_depth=True`, while the decoded video-token Gaussians
are not guaranteed to be depth-sorted. That can skip the renderer's internal
sort, change alpha compositing order, and weaken front-to-back early-out. The
wrapper default has since been changed back to `False`, so this run should be
treated as suggestive but not a clean final baseline.

## Result

Completed 250 steps on MPS in about 573 seconds.

Final summary:

```text
Loss                         0.1367368
Loss/Reconstruction          0.1365391
Eval/Loss                    0.1444065
Eval/L1                      0.0949012
Eval/SSIM                    0.3151448
Eval/PSNR                    16.731864
Eval/TemporalToFirstL1Ratio  0.4940645
Eval/TemporalAdjacentL1Ratio 0.0446135
Camera/EvalFOVDegrees        57.9260
Camera/EvalRadius            2.8184
Camera/EvalRotDeltaMeanDeg   4.4278
Camera/EvalTransDeltaMean    0.5662
```

Compared with the earlier 128px local implicit-camera run `h5lefdtf`:

```text
Eval/Loss: 0.14816 -> 0.14441
Eval/L1:   0.09686 -> 0.09490
Eval/SSIM: 0.29336 -> 0.31514
Temporal-to-first L1 ratio: 0.33698 -> 0.49406
Adjacent L1 ratio:          0.02457 -> 0.04461
```

The final preview visually recovers a more object-shaped region than the weak
background-only failure, but the dog is still blurred and low-detail. The
improvement supports the init-health hypothesis, but the renderer sorting caveat
means the same ablation should be rerun under the corrected fast-mac default
before treating the numbers as clean.

## Current Interpretation

Supported:

- RGB/token initialization mattered. This was not only a V-JEPA/local-encoder
  question.
- The prior 128px failures included an init/capacity/loss-contract component,
  not just a renderer or pretrained-backbone failure.
- Stronger init produces more temporal variation than the earlier local/V-JEPA
  128px smoke runs.

Still unresolved:

- Whether the better shape is due mostly to RGB uniform bias init or the larger
  query/head scale. The combined ablation tested both at once.
- Whether 128px is now source/model limited or still training-limited.
- Whether V-JEPA helps once the same init improvements and 256px end-to-end
  raster contract are applied.

## Next Tests

1. Add/re-run with step-0 logging so initialization media is visible.
2. Split the combined ablation into:
   - RGB uniform only.
   - strong query/head scale only.
   - combined with slightly lower scale.
3. Run the same init contract at 256px before judging dog detail.
4. Apply the init contract to the V-JEPA fpc16/256 comparison so backbone
   comparisons are not confounded by different initialization health.
