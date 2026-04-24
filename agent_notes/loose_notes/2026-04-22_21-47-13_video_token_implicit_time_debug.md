# Video-token implicit time debug

## Context

The implicit video-token trainer showed a suspicious failure mode: generated
validation videos reused nearly the same rendered frame across time. The goal
was to check whether the model/trainer actually knew the timestamp for each
frame in a batch, then fix it and add a debug metric for per-frame similarity.

## What was wrong

- `train_video_token_implicit_dynamic.py` normalized every sampled training
  window to local clip time `0..1`.
  - A window covering frames `0..15` and a window covering frames `30..45`
    therefore presented the same local times to the model.
  - This destroyed absolute sequence time and encouraged repeated temporal
    solutions.
- `DynamicVideoTokenGSImplicitCamera` only added `time_proj(...)` after
  video-token cross-attention.
  - The query decoder saw the same queries for every requested frame.
  - Time could only perturb already-refined tokens late, which made it too easy
    for the model to collapse to one mostly static state.

## Fixes made

- `prepare_clip(...)` now passes `sequence_data.frame_times[clip_indices]`
  directly, preserving absolute frame times from the loaded sequence.
- `DynamicVideoTokenGSImplicitCamera.forward(...)` now encodes the video once
  and refines queries separately for each requested `decode_time`.
  - `time_proj(decode_time)` is injected into non-global query tokens before
    query-to-video cross-attention.
  - The global camera token is left un-timed so the learned global camera stays
    sequence-global.
- Added temporal eval/debug metrics:
  - `Eval/TemporalPredAdjacentL1`
  - `Eval/TemporalGTAdjacentL1`
  - `Eval/TemporalAdjacentL1Ratio`
  - `Eval/TemporalPredToFirstL1`
  - `Eval/TemporalGTToFirstL1`
  - `Eval/TemporalToFirstL1Ratio`
  - `Eval/TemporalPredAdjacentSSIM`
  - `Eval/TemporalGTAdjacentSSIM`

## Verification

- `uv run python -m py_compile` passed for the modified trainer/model files.
- Time probe after the trainer fix showed absolute times:
  - frames `0..15`: `0.0..0.333`
  - frames `16..31`: `0.356..0.689`
  - frames `30..45`: `0.667..1.0`

## Runs

### Absolute-time trainer fix only

- W&B `fn120nki`: https://wandb.ai/nbardy/dynaworld/runs/fn120nki
- 100 steps, 128px/4fps, 8192 splats, fast-mac batch render.
- Full eval:
  - `Eval/Loss 0.17578`
  - `Eval/L1 0.12398`
  - `Eval/SSIM 0.23407`
  - `Eval/PSNR 15.68`
- Temporal debug:
  - `Eval/TemporalPredAdjacentL1 0.000735`
  - `Eval/TemporalGTAdjacentL1 0.086187`
  - `Eval/TemporalAdjacentL1Ratio 0.00853`
  - `Eval/TemporalPredToFirstL1 0.01655`
  - `Eval/TemporalGTToFirstL1 0.12723`
  - `Eval/TemporalToFirstL1Ratio 0.13006`
  - `Eval/TemporalPredAdjacentSSIM 0.99982`
  - `Eval/TemporalGTAdjacentSSIM 0.45444`
- Interpretation: the trainer time bug was real but not sufficient; output was
  still almost static.

### Time-conditioned query cross-attention

- W&B `qgjwozbn`: https://wandb.ai/nbardy/dynaworld/runs/qgjwozbn
- 100 steps with the architecture patch.
- Full eval:
  - `Eval/Loss 0.15101`
  - `Eval/L1 0.09978`
  - `Eval/MSE 0.02287`
  - `Eval/SSIM 0.28808`
  - `Eval/PSNR 16.41`
- Temporal debug:
  - `Eval/TemporalPredAdjacentL1 0.001498`
  - `Eval/TemporalGTAdjacentL1 0.086187`
  - `Eval/TemporalAdjacentL1Ratio 0.01738`
  - `Eval/TemporalPredToFirstL1 0.03495`
  - `Eval/TemporalGTToFirstL1 0.12723`
  - `Eval/TemporalToFirstL1Ratio 0.27474`
  - `Eval/TemporalPredAdjacentSSIM 0.99797`
  - `Eval/TemporalGTAdjacentSSIM 0.45444`

### 500-step probe after architecture patch

- W&B `gypzvibd`: https://wandb.ai/nbardy/dynaworld/runs/gypzvibd
- Laptop suspended twice during the run, but the run reached step 500 and
  wrote final eval.
- Full eval:
  - `Eval/Loss 0.14551`
  - `Eval/L1 0.09501`
  - `Eval/MSE 0.02108`
  - `Eval/SSIM 0.30493`
  - `Eval/PSNR 16.76`
- Temporal debug:
  - `Eval/TemporalPredAdjacentL1 0.003216`
  - `Eval/TemporalGTAdjacentL1 0.086187`
  - `Eval/TemporalAdjacentL1Ratio 0.03731`
  - `Eval/TemporalPredToFirstL1 0.05827`
  - `Eval/TemporalGTToFirstL1 0.12723`
  - `Eval/TemporalToFirstL1Ratio 0.45799`
  - `Eval/TemporalPredAdjacentSSIM 0.99413`
  - `Eval/TemporalGTAdjacentSSIM 0.45444`

## Interpretation

The original single-frame reuse issue had two concrete causes: local time
renormalization in the trainer and time arriving too late in the model. Both
are patched. The new metrics show the architecture now uses time more than
before, but it is still temporally underactive: adjacent-frame predicted change
is only about `3.7%` of GT after 500 steps, even though accumulated change from
the first frame reaches about `46%` of GT. This is no longer the exact same
single-frame collapse, but it is not a solved temporal baseline.
