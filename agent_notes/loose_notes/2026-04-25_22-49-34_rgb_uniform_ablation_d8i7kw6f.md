# RGB-Uniform Strong-Init Ablation Result d8i7kw6f

## Context

The user ran:

```bash
PYTHONPATH=src/train uv run python src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_ablate_init_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc
```

W&B run:

```text
https://wandb.ai/nbardy/dynaworld/runs/d8i7kw6f
```

Local W&B root:

```text
wandb/run-20260425_223346-d8i7kw6f/
```

This was a single-video 128px implicit-camera video-token overfit run, not the
new 20/10 scene-distinct dataset baseline. It used:

- source: `test_data/test_video_small_128_4fps.mp4`
- frame count: 46
- train frame window: 16
- local video encoder
- implicit learned camera path
- fast-mac renderer
- 128 tokens x 64 gaussians/token = 8192 gaussians
- `rgb_init="uniform"`, `rgb_init_min=0.0`, `rgb_init_max=1.0`
- `query_token_init_std=0.8`
- `head_output_init_std=0.12`
- 250 steps

## Observed Result

Final summary from local `wandb-summary.json`:

```text
Loss/Reconstruction: 0.136539
Loss: 0.136737
Eval/Loss: 0.144406
Eval/L1: 0.094901
Eval/MSE: 0.021223
Eval/SSIM: 0.315145
Eval/PSNR: 16.7319
Camera/EvalFOVDegrees: 57.9260
Camera/EvalRadius: 2.8184
Camera/EvalRotationDeltaMeanDegrees: 4.4278
Camera/EvalTranslationDeltaMean: 0.5662
Eval/TemporalPredAdjacentL1: 0.003845
Eval/TemporalGTAdjacentL1: 0.086187
Eval/TemporalAdjacentL1Ratio: 0.04461
Eval/TemporalPredToFirstL1: 0.062857
Eval/TemporalGTToFirstL1: 0.127225
Eval/TemporalToFirstL1Ratio: 0.4941
```

Media:

```text
wandb/run-20260425_223346-d8i7kw6f/files/media/images/Render_GT_vs_Pred_240_dd078866dfc3eb275f32.png
wandb/run-20260425_223346-d8i7kw6f/files/media/videos/Render_GT_Video_240_b1540aca9cbd7a80e956.mp4
```

Qualitatively, the final preview learns a plausible foreground animal/blob and
background structure. It is still blurry and not a faithful dynamic
reconstruction, but it is not the old blank/gray failure mode.

## Hypothesis Status

Hypothesis: weak RGB/diversity initialization was one bottleneck.

Status: partially supported.

Evidence:

- The prior init diagnostics measured RGB collapse around gray and weak
  same-split token diversity.
- This run used the RGB-uniform plus stronger token/head init.
- The final media visibly has more useful color/shape structure than the
  background-only failure pattern.

What keeps this from being confirmed:

- This is not yet a paired rerun against the unchanged baseline with identical
  logging and run length.
- It is still a single-video overfit target, so it does not say anything about
  scene generalization.

Hypothesis: the 128px local video-token implicit model can overfit a single
video enough to learn coarse shape.

Status: supported, with limits.

Evidence:

- The final render has a recognizable coarse scene/object layout.
- Eval loss is finite and stable.
- Camera radius/FOV remain plausible rather than collapsing to an obviously
  degenerate value.

Limit:

- The quality is not close to high-fidelity reconstruction; SSIM is only about
  `0.315`.

Hypothesis: temporal dynamics remain underfit or over-smoothed.

Status: supported.

Evidence:

- `Eval/TemporalPredAdjacentL1` is `0.003845` while GT adjacent L1 is
  `0.086187`.
- The adjacent temporal ratio is only `0.0446`.
- Predicted frames are much more similar to each other than the real video.

Interpretation:

The model can form a coarse quasi-static splat scene and move/view it somewhat,
but it is not yet capturing the target video motion at frame-to-frame scale.

Hypothesis: V-JEPA is not the only immediate bottleneck.

Status: strengthened.

Evidence:

- A local encoder plus better initialization already improved the qualitative
  single-video fit.
- That means model/init/loss/camera path issues still matter before we can
  attribute failures to the video backbone.

## Contribution To The Thread

This run sits between the init-diagnostics work and the 256px V-JEPA baseline
work. It says:

1. The model was not purely blocked by renderer plumbing; the run trains.
2. Initialization quality affects whether the model reaches a recognizable
   foreground/shape regime.
3. Coarse shape learning is easier than temporal fidelity.
4. The next fair V-JEPA comparison should inherit the healthier init/logging
   setup or at least compare against it as a local-encoder control.

It does not answer:

- whether V-JEPA is better than the local encoder,
- whether the model generalizes across the 20/10 scene-distinct split,
- whether novel-view GT validation works,
- or whether camera specs/adapters are correct.

## Next Tests

1. Rerun the unchanged 128px baseline with the same logging cadence and compare
   media/metrics against `d8i7kw6f`.

2. Add step-0 render/video logging so future runs can distinguish good init,
   bad init, optimization collapse, and slow convergence.

3. Run the new 256px scene-distinct local encoder config, then the matched
   frozen V-JEPA config:

```bash
./src/train_scripts/train_local_mac_30_clip_vjepa2_256_baseline.sh local
./src/train_scripts/train_local_mac_30_clip_vjepa2_256_baseline.sh vjepa
```

4. Add foreground/motion-sensitive diagnostics. Full-frame sky/grass-heavy loss
   can hide object failure.

5. Test a temporal-motion ablation. The current run fits coarse appearance but
   temporal adjacent motion is much too low.

6. Continue the GT validation path separately: DeepView `CameraSpec` adapter,
   then camera-ready metric runner.
