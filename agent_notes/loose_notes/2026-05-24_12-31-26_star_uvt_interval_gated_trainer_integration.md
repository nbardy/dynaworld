# STAR UVT Interval-Gated Trainer Integration

Worker B lane: wire the existing interval-gated q-UVT bridge into the real
trainer path without touching nonlinear/projective Metal evaluator internals.

Changes made:

- Added `render_uvt_tubes_metal_interval_gated_backward(...)` in the STAR UVT
  trainer harness. It uses native `render_gated` for forward and native
  `direct_atomic_backward_gated` for VJP.
- Added `full_active_intervals(...)` so the ordinary source-view trainer can
  select the gated backend even when its current screen-time model has a single
  whole-video gauge domain.
- Added `metal_tile_interval_gated` as a selectable
  `uvt.render_backend` in `video_fit_comparison.py`.
- Added `validate_uvt_backend_modes(...)` so the new backend explicitly requires
  `reduction_mode=index_add` and `sample_emission_mode=direct_atomic`, matching
  the native `direct_atomic_gated` path.
- Added `tests/test_star_uvt_trainer_interval_gated.py`.

Why this is scoped correctly:

- The nonlinear/projective evaluator internals were not changed.
- The new backend is trainer-dispatch plumbing plus a trainer-harness autograd
  wrapper.
- The wrapper accepts explicit `active_start`/`active_stop` tensors, so later
  projective/gauge-domain bridges can pass real chart-domain intervals instead
  of leaking outside their certified windows.
- The current real trainer path uses full intervals, which is the correct
  degenerate domain for ordinary STAR UVT screen-time tubes.

Verification:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_trainer_interval_gated.py \
  tests/test_star_uvt_projective_correctness.py -q
```

Result: `14 passed in 14.38s`.

After a tiny cleanup, the new focused test alone was rerun:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_trainer_interval_gated.py -q
```

Result: `2 passed in 10.97s`.

Real trainer smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train.py /tmp/star_uvt_interval_gated_smoke.jsonc
```

The temp config was derived from
`src/train_configs/star_uvt_rgb_testvideo_64f_512_directatomic_8192t_20step_media.jsonc`
with `target_size=64`, `max_frames=4`, `tube_count=128`, `steps=1`,
`wandb_enabled=false`, no media, and:

```json
"render_backend": "metal_tile_interval_gated",
"reduction_mode": "index_add",
"sample_emission_mode": "direct_atomic"
```

Result: passed. The row reported
`render_backend=metal_tile_interval_gated`, `initial_loss=0.15689826011657715`,
`final_loss=0.12637178599834442`, `loss_ratio=0.8054377779871413`.

Next step:

- Let Worker A own nonlinear/projective Metal evaluator internals.
- Worker B can next wire a real projective/gauge-domain bridge producer into a
  trainer-adjacent path that supplies nontrivial `active_start`/`active_stop`
  intervals, now that the gated trainer backend itself is selectable and smoked.
