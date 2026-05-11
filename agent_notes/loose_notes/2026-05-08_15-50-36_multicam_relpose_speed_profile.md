# Multicam Relpose Speed Profile

Session goal: explain why the latest multicam relative-pose feature-splatting run no longer feels like the old 20s-1m loop, and get component timings rather than blaming the rasterizer blindly.

Completed run inspected:

- Log: `outputs/run_logs/20260508_144522_fast512_tokenbudget_train.log`
- Config: `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_multires64_128_256_512_tokenbudget_world4_fast_16f_8192splats_goodset_train0006_0014_holdout0005_alpha1_128_relpose_outputinit012.jsonc`
- W&B: `https://wandb.ai/nbardy/dynaworld/runs/24absic1`
- 250 tqdm steps took `1507s` (`25.12m`, `6.03s/step` average). Progress chunks were spiky: median about `3s/step`, p90 about `11s/step`, worst chunks `44-77s`.

Profiler added:

- `train.profile_timing`: opt-in section timer.
- `train.profile_timing_sync`: sync CUDA/MPS around sections for real phase attribution.
- `train.profile_timing_log_every`: print cadence.
- Trainer logs `Timing/..._s` scalar keys and prints `Timing step N: ...`.
- Sections include sample, source decode, relpose feature memory/head/camera math, render rasterize/colorize/compose, reconstruction loss, backward, optimizer, and step total. Nested sections intentionally do not sum to `step_total`.

Saved local profiles:

- Random schedule, 1 warmup + 5 measured: `outputs/benchmarks/multicam_relpose_fast512_tokenbudget_step_profile_20260508.json`
- Forced one measured step per resolution: `outputs/benchmarks/multicam_relpose_fast512_tokenbudget_resolution_profile_20260508.json`

Forced-resolution timings on local MPS, W&B disabled, no validation/media:

| render | detail | decoded tokens | step total | raster | colorize | recon loss | backward | source decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0 | 72 | `0.663s` | `0.068s` | `0.005s` | `0.006s` | `0.258s` | `0.192s` |
| 128 | 0 | 72 | `1.071s` | `0.145s` | `0.019s` | `0.028s` | `0.457s` | `0.250s` |
| 256 | 1 | 104 | `2.217s` | `0.213s` | `0.050s` | `0.073s` | `1.161s` | `0.398s` |
| 512 | 2 | 128 | `10.253s` | `1.221s` | `0.806s` | `0.704s` | `6.811s` | `0.333s` |

Read:

- The real low-res training loop is still sub-second to about one second. The latest schedule is slow because the 512px tail is very expensive and because the full W&B run includes logging/media/system stalls that are not visible in raw optimizer-step timing.
- Weighted by the configured probabilities `[64:25%, 128:45%, 256:25%, 512:5%]`, the forced-resolution rows imply roughly `1.7s/step` before validation/media/checkpointing. That is about `7m` for 250 steps, not the `25m` observed in the W&B log, so online logging/media/MPS contention likely explains the rest.
- The 512px step is dominated by backward (`6.8s`), then render/colorize/loss. Raster alone is not the whole story, but render/loss creates the large tensors whose backward dominates.
- Relpose feature memory is not the bottleneck (`~0.003-0.004s`), and relpose head itself is small (`~0.02-0.04s`). Source decode is a steady `0.19-0.40s`.

Immediate levers:

- For rapid iteration, cap training at 256px or make 512 eval-only until quality proves the 5% tail is worth the wall time.
- If keeping 512 in train, reduce 512 probability below 5%, sample fewer camera-swap pairs per step, or try render/loss microbatching for the 512 branch.
- Use `profile_timing=true` on future W&B runs when changing renderer, resolution schedule, camera-swap pairs, or token budget; otherwise the full run wall time hides whether the slowdown is optimizer, render/loss backward, validation media, or logging.
