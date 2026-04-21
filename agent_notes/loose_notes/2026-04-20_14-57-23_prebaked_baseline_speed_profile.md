# Prebaked Baseline Speed Profile

User asked why the known-camera/prebaked dynamic baseline is around 1.6 it/s
and whether it is doing a full all-frame batch each step.

Current config path:

- `src/train_configs/local_mac_overfit_prebaked_camera.jsonc`

Important config values:

- `model.size = 32`, so this path trains/renders/losses at 32x32.
- `train.frames_per_step = 0`, which `dynamicTokenGS.py` maps to all loaded
  frames. The current test sequence loads 23 frames.
- `logging.image_log_every = 25` and `logging.video_log_every = 25`, so every
  25 steps does extra full-sequence validation rendering.

Benchmark run from a custom `uv run python` snippet with W&B omitted and MPS
synchronization around each timed phase:

```text
device mps
frames 23 size 32 gaussians 512 renderer dense

frames_per_step 4
  prep                     0.36 ms
  model_forward            9.20 ms
  render_loss_forward     44.12 ms
  backward_step           85.39 ms
  total                  139.08 ms
  approx it/s 7.19

frames_per_step 23
  prep                     0.36 ms
  model_forward           23.77 ms
  render_loss_forward    169.93 ms
  backward_step          365.97 ms
  total                  560.03 ms
  approx it/s 1.79
```

AMP/flash-attn check was slower on this Mac/MPS path:

```text
amp False frames 23: 562.83 ms, 1.78 it/s
amp True  frames 23: 634.37 ms, 1.58 it/s
```

Conclusion: the bottleneck is not input resolution and not mostly attention.
Most wall time is dense PyTorch splat rendering and the backward pass through
23 per-frame render graphs. Reducing `frames_per_step` is the fastest config
knob. The best code-level candidate is a batched dense renderer for `[B, G]`
frames/cameras instead of the current Python loop over frames.

Follow-up implementation in the same session added batched dense projection and
rendering for `GaussianSequence` payloads, then routed the known-camera
`dynamicTokenGS.py` trainer through it for dense mode. A synthetic batch matched
the old per-frame dense renderer within float tolerance:

```text
max_diff 1.1920928955078125e-07
```

Post-change timing with W&B omitted and MPS synchronization:

```text
frames 4:  92.75 ms, 10.78 it/s, frame/s 43.1
frames 23: 427.64 ms,  2.34 it/s, frame/s 53.8
```

One-step offline W&B smokes passed for:

- `local_mac_overfit_prebaked_camera.jsonc`: 23 frames per step, loss about `0.2046`
- `local_mac_overfit_prebaked_camera_64_4fps.jsonc`: 4 frames per step, loss about `0.1539`
- `local_mac_overfit_prebaked_camera_128_4fps.jsonc`: 4 frames per step, loss about `0.3596`

Follow-up config cleanup added `data.video_variant` for prebaked configs:

- `small_32_2fps` -> `test_data/dust3r_outputs/test_video_small_all_frames`, model size 32
- `small_64_4fps` -> `test_data/dust3r_outputs/test_video_small_64_4fps_all_frames`, model size 64
- `small_128_4fps` -> `test_data/dust3r_outputs/test_video_small_128_4fps_all_frames`, model size 128

`data.sequence_dir` can still be set to an explicit path to override the
variant mapping. `dynamicTokenGS.resolve_config(...)` fails loudly if a known
variant's expected `model.size` does not match the config.

Observed failure after the first 64px W&B run: `lr=0.005` on
`small_64_4fps` collapsed to a nearly constant gray render around loss
`0.14`. Renderer batch-vs-loop forward and gradient checks matched, so this was
not a batched renderer math regression. The 64/4fps DUSt3R bake has much
narrower normalized focal (`fx/width ~= 2.73`) than the original 32/2fps bake
(`fx/width ~= 1.77`), and the higher LR drives the model into a mean-color
basin. Short probes on the 64 config:

```text
lr=0.005 step 200 loss 0.1711 render_std 0.0192
lr=0.001 step  50 loss 0.0887 render_std 0.1349
lr=0.001 step 100 loss 0.1061 render_std 0.1816
```

Updated the 64/4fps and 128/4fps configs to `train.lr = 0.001`.
