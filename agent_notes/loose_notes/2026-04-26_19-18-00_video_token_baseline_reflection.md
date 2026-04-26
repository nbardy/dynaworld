# Video-Token Baseline Reflection

## What Has Shaped Up

The current best tiny local baseline is:

- 16 loaded frames from `test_data/test_video_small_128_4fps.mp4`
- 128px render/input
- 8192 splats
- fast-mac renderer
- implicit learned orbit/path camera
- 96 static splat tokens + 32 dynamic splat tokens
- 4 query cross-attention layers
- RGB-uniform stronger query/head init
- precomputed V-JEPA 2.1 ViT-B/384 tokens

Best run so far:

```text
https://wandb.ai/nbardy/dynaworld/runs/oaor6um2
```

The 250-step metrics were:

```text
Eval/Loss 0.0881
Eval/L1   0.0615
Eval/SSIM 0.6109
Eval/PSNR 20.29
TemporalPredAdjacent / GTAdjacent 0.6322
```

A cached longer follow-up was also run and interrupted after the step-500 video
checkpoint:

```text
https://wandb.ai/nbardy/dynaworld/runs/mybv736f
```

Latest synced metrics:

```text
Eval/Loss 0.0547
Eval/L1   0.0413
Eval/SSIM 0.7836
Eval/PSNR 23.69
TemporalPredAdjacent / GTAdjacent 0.8009
```

This is the best looking fit at this scale so far by a wide margin.

## What Worked

Static/dynamic capacity split was the major architectural turn. Plainly adding
more cross-attention layers made learned-time outputs more frozen and worse,
while the 96/32 split gave the model a direct low-rank dynamic bank without
forcing every token to move.

V-JEPA features then gave the split a much better conditioning signal. The
feature run beat the local-encoder split strongly on reconstruction and SSIM:

```text
local split:    Eval/Loss 0.1195, SSIM 0.4287, pred/GT adjacent 0.3408
V-JEPA split:   Eval/Loss 0.0881, SSIM 0.6109, pred/GT adjacent 0.6322
```

The base architecture now has a credible small-scale shape:

```text
video features -> world/query tokens -> static/dynamic splat bank -> differentiable render
```

## What Was Smarter Than Expected

The diagnostics mattered. Pixel loss alone would have hidden whether the model
was moving splats, camera, or neither. The decoded XYZ/RGB/opacity and camera
adjacent metrics made it clear that:

- cross-attn4 learned-time froze splats
- sinusoidal time moved things but also moved camera
- static/dynamic split used dynamic capacity without huge camera motion
- V-JEPA split moved both splats and camera more, which is useful but needs
  validation beyond same-source overfit

The feature cache boundary also mattered. The trainer prebakes full loaded
`SequenceData`, not sampled clips. Capping this fast baseline to 16 loaded
frames made it local-Mac viable.

After this run, `VideoFeatureCache` was patched to instantiate feature
extractors lazily. Previously even a cache-hit run loaded the V-JEPA torchhub
model before discovering the `.pt` existed; future cache-hit runs should avoid
that startup cost.

## What Could Be Smarter

The current precomputed feature path is not clip-aware. If `data.max_frames=0`
on a longer video, V-JEPA bakes the whole loaded sequence. That breaks the
mental model of "train 16 frames" and makes runtime/memory scale badly.

The query decoder repeats cross-attention over a 4608-token V-JEPA memory,
including per-frame camera refinement in the static/dynamic split. That is why
cached-feature training is slower than the local-encoder split despite not
running V-JEPA in the optimizer loop.

We still need held-out and multi-camera validation. This is the best local
single-clip fit, not yet proof of generalization or novel-angle correctness.

## Next Bets

1. Train the same config longer from the already-written feature cache.
2. Measure pure cached train-loop runtime separately from first-bake overhead.
   The first cached long run reached strong quality but had variable runtime
   because of system load; repeat when the Mac is quiet if runtime matters.
3. Add clip-aware feature caching or a feature temporal downsampler before
   using this on longer source videos.
4. Move the static/dynamic + feature setup onto scene-distinct train/test clips.
5. Evaluate on multi-camera validation, where source-time novel-angle GT exists.
