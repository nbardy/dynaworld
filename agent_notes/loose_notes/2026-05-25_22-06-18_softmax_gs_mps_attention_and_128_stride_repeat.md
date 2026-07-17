# Softmax-GS MPS Attention Fix And 128px Stride Repeat

Date:
    2026-05-25 22:06 local

Context:
    The Softmax-GS short-term plan said the next blocker was scale/repeat:
    128px or 8f rows aborted before rasterization with an MPS
    `sliceDimension` assertion. We needed to determine whether this was a
    Softmax-GS renderer problem, a model-forward problem, or a config/data
    shape problem.

What changed:
    Added an MPS-safe manual batch-first MHA fallback in
    `src/train/gs_models/dynamic_video_token_gs_implicit_camera.py`.
    `QueryCrossAttentionBlock` still uses PyTorch `nn.MultiheadAttention` by
    default, but switches to the manual path on MPS when cross-attention memory
    exceeds 32,768 tokens.

Why:
    A synthetic repro showed MPS `nn.MultiheadAttention` with `B=1`, `Lq=10`,
    `E=64`, `H=4` succeeds at memory length 32,768 and crashes at 40,960 with
    the same assertion. The prior safe 64px/4f RGB-pyramid config has 20,480
    memory tokens. The failing 64px/8f config has 40,960, 128px/4f has 81,920,
    and 128px/16f has 327,680.

Validation:
    `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_mps_safe_cross_attention.py -q`
    passed with `2 passed`. The test covers CPU parity against
    `nn.MultiheadAttention` and a 40,960-token MPS smoke that would have hit
    the backend assertion without the fallback.

    A 128px/16f enabled forward/tape smoke also completed:
    `research_experiments/softmax_gs/diagnose_tape_coverage.py ... --train-steps 0 --k-values 16 --views train0`.
    This proves the old blocker is not the renderer.

Unstrided 128px/16f attempt:
    `local_mac_multicam_softmax_gs_noop_rgb_pyramid_128_16f_512splats_20step.jsonc`
    no longer crashes, but the full-memory route is locally impractical.
    It was interrupted after 3/20 steps at 9:47. Partial offline W&B:
    `wandb/offline-run-20260525_214838-pjxwtb2c`.

Practical stride16 configs:
    Added:
    `src/train_configs/local_mac_multicam_softmax_gs_noop_rgb_pyramid_128_16f_512splats_stride16_20step.jsonc`
    and
    `src/train_configs/local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_128_16f_512splats_stride16_20step.jsonc`.
    These keep 128px render, 16 frames, and 512 splats, but set
    `model.video_feature_token_stride=16`, reducing RGB-pyramid memory to the
    proven 20,480-token envelope.

Results:
    No-op stride16:
        initial/final train loss `0.5843 -> 0.2577`
        train0 `10.9996/0.1416`
        train1 `12.2710/0.1729`
        heldout camera_0040 `12.1234/0.1244`
        step20 total/backward/raster `1865/336/122ms`
        offline run `wandb/offline-run-20260525_220100-zod704i9`

    Enabled K=16 stride16:
        initial/final train loss `0.5843 -> 0.2504`
        train0 `10.8973/0.1372`
        train1 `11.6462/0.1581`
        heldout camera_0040 `12.2092/0.1088`
        step20 total/backward/raster `1107/197/65ms`
        offline run `wandb/offline-run-20260525_220309-pkrvtzda`

Interpretation:
    The scale repeat is mixed, not a Softmax-GS promotion. Enabled K=16 gets a
    tiny heldout-PSNR nudge (`+0.0858dB`) and slightly better final train loss,
    but loses heldout SSIM and both train-view metrics. Combined with the
    64px/4f/512 repeat losing heldout PSNR, the tiny 128-splat heldout jump is
    not cleanly repeated.

Decision implication:
    Do not port Softmax-GS to STAR UVT or WorldFoam from this evidence. The
    remaining Softmax-only fork worth trying is learned per-Gaussian/per-layer
    Softmax parameters in dynamic GS; otherwise return effort to STAR support
    and WorldFoam challenger gates.
