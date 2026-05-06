# Video Token Implicit Camera Follow-Ups

## Context

The video-token implicit-camera trainer now runs from JSONC configs and separates
encoder input size from render/loss viewport size.

Current default configs:

- full: `src/train_configs/local_mac_overfit_video_token_full.jsonc`
- smoke: `src/train_configs/local_mac_overfit_video_token_smoke.jsonc`

The full config intentionally uses `model.size = 384` and `render.render_size =
192` to keep MPS memory practical while still feeding the video encoder the
larger clip.

## Follow-Up TODOs

1. Decide whether the "full" baseline should keep `render_size = 192` as the
   default or move back to `384` once renderer memory is improved.
2. Add a small invariant check for `camera_for_viewport(...)`: scaling
   `fx/fy/cx/cy` from source to target viewport should preserve normalized ray
   coordinates and should not change `camera_to_world`.
3. Add a config option for selective frame supervision: encode the full clip but
   render/loss only `K` sampled decode times per step.
4. Benchmark the full config across render sizes `128`, `192`, and `384` with
   the same seed/run settings so render-size tradeoffs are visible in W&B.
5. Design a shared conditioning contract for single images and videos. The
   target shape is one renderer/API that can consume `image_or_video ->
   world_tokens, source_camera_token` and render from a query camera, but this
   requires architecture changes around `T=1` inputs, dynamic-token behavior, and
   `query_time`; see `research_notes/training_ideas_for_novel_synthesis.md`.
6. For mixed multi-camera pretraining, continue from the source-anchored
   camera-swap path. `train.camera_swap_mode="oracle_relative"` renders
   `W_source + calibrated Delta_source_to_target -> target`, and
   `train.camera_swap_mode="learned_residual"` adds the tiny
   `relpose(F_source, F_target) -> residual SE(3)` head with target features
   blocked from the world decoder. Residual identity and cycle losses are wired.
   The first 250-step V-JEPA train/eval run is recorded in `BASELINES.md`.
   Remaining work: leakage probes, a longer/seeded rerun, and a query-only
   inference path for camera requests that do not have target RGB.

## Guardrails

- Do not reintroduce env-var fanout for every hyperparameter.
- Do not add a giant Python `DEFAULT_CONFIG` mirror.
- Keep shell scripts as thin config launchers.
- Treat `render_size` as a viewport resize, not a crop.
