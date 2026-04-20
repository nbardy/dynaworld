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

## Guardrails

- Do not reintroduce env-var fanout for every hyperparameter.
- Do not add a giant Python `DEFAULT_CONFIG` mirror.
- Keep shell scripts as thin config launchers.
- Treat `render_size` as a viewport resize, not a crop.
