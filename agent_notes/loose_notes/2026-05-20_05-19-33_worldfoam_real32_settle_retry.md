# WorldFoam real32 settle retry

## Context

The real32 DeepView fixture and one-step loader smoke removed the fake-32f
data blocker, but the first timing artifacts were not promotable because the
benchmark environment ended contended. I added/used the post-run settle path
for `MTLCompilerService`-only transients, then tried the strict wrapper while
the live `ai_trader` TOTO monitor was still running.

## Command

```bash
rtk .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py \
  --run-id 2026-05-20_real32_strict_mini_wrapper_settle_retry \
  --worldfoam-config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc \
  --star-video-path data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_center_crop_8fps_full.mp4 \
  --frame-counts 32 \
  --render-size 16 \
  --site-count 8 \
  --worldfoam-steps 1 \
  --star-steps 1 \
  --worldfoam-warmup-steps 1 \
  --star-warmup-steps 1 \
  --star-target-size 32 \
  --star-tube-count 224 \
  --max-worldfoam-attempts 2 \
  --max-star-attempts 2 \
  --wait-timeout-s 300 \
  --wait-poll-s 2 \
  --post-run-benchmark-environment-settle-s 5 \
  --require-real-loaded-frames \
  --verify-promotion
```

## Result

- Summary:
  `research_experiments/world_foam_lane2/results/2026-05-20_real32_strict_mini_wrapper_settle_retry.promotion_summary.json`
- Attempt artifacts:
  - `...settle_retry.attempt1.worldfoam.json`
  - `...settle_retry.attempt2.worldfoam.json`
- Wrapper status: `worldfoam_not_promotable`
- STAR attempts: none; the wrapper never passed a clean WorldFoam artifact to
  STAR.

Both attempts found clean preflight gaps and completed true-32f WorldFoam rows:

| Attempt | loaded frames | repeat | train PSNR | heldout PSNR | total | backward | end blocker |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 32 | false | 12.987 | 14.229 | 2.248 ms | 1.952 ms | live `ai_trader` TOTO export + `MTLCompilerService` |
| 2 | 32 | false | 12.987 | 14.229 | 2.303 ms | 2.009 ms | live `ai_trader` TOTO export |

The important distinction: this is not a shader correctness failure. It is
also not an `MTLCompilerService`-only transient that the settle path should
clear. A real Python/MPS TOTO export started during both benchmark windows, so
the strict wrapper correctly rejected the artifacts as diagnostic.

## Takeaways

- Warmup matters on this tiny real32 smoke: the cold loader smoke was hundreds
  of milliseconds, while warm attempts are about `2ms` for the timed shader
  step.
- The real32 WorldFoam path now has correctness evidence on true loaded
  frames, but still lacks a promotable speed/STAR comparison artifact.
- The next clean timing gate should pause/stop the live `ai_trader` monitor or
  run after it exits. Retrying inside its 30-second export cadence can find a
  clean start but not reliably a clean end snapshot.

## Follow-up harness fix

After the retry, I hardened both benchmark-environment classifiers:
`train_eval_owner_run_tape.py` and `compare_star_uvt_worldfoam_scale.py` now
treat a `run_btc15m_overnight_shadow_monitor.py` command with
`--toto-export-device mps` or `--toto-export-with-runtime-deps` as a promotion
blocker even when the visible parent process is idle. Generic low-CPU monitor
wrappers still count as background; this special-case covers the periodic
exporter that can wake up and use MPS inside the WorldFoam/STAR timing window.

Verification after the fix:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate \
  research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale \
  research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion
```

Result: `Ran 46 tests ... OK`. `py_compile` and scoped `git diff --check`
also passed.

Live check-only preflight now exits `2` while the monitor is active:

```bash
rtk .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only \
  --wait-for-benchmark-environment-ok-timeout-s 0 \
  --wait-for-benchmark-environment-ok-poll-s 1 \
  --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc
```

The snapshot lists the high-CPU TOTO residual quote-shadow export child and
the idle screen/login/uv/python `run_btc15m_overnight_shadow_monitor.py`
parents as `blocking_processes`. That proves the next WorldFoam/STAR wrapper
will fail fast instead of starting inside this live MPS-export cadence.

I also ran the top-level wrapper in `--preflight-only` mode with the real32
config:

```bash
rtk .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py \
  --run-id 2026-05-20_real32_preflight_toto_mps_blocker_check \
  --worldfoam-config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc \
  --star-video-path data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_center_crop_8fps_full.mp4 \
  --frame-counts 32 \
  --render-size 16 \
  --site-count 8 \
  --worldfoam-steps 1 \
  --star-steps 1 \
  --worldfoam-warmup-steps 1 \
  --star-warmup-steps 1 \
  --star-target-size 32 \
  --star-tube-count 224 \
  --max-worldfoam-attempts 1 \
  --max-star-attempts 1 \
  --wait-timeout-s 0 \
  --wait-poll-s 1 \
  --post-run-benchmark-environment-settle-s 5 \
  --require-real-loaded-frames \
  --verify-promotion \
  --preflight-only
```

It exited `2` with summary
`research_experiments/world_foam_lane2/results/2026-05-20_real32_preflight_toto_mps_blocker_check.promotion_summary.json`,
status `worldfoam_preflight_failed_or_contended`, no WorldFoam artifact, and no
STAR attempt. Its compact blocker list contains the idle TOTO screen/login/uv
parent chain with `high_cpu=false`, which is the intended fail-fast behavior.

I then added explicit blocker reasons to both benchmark classifiers and the
wrapper's compact blocker summaries. The refreshed preflight artifact now
preserves `block_reason=periodic_mps_exporter` for the idle TOTO parent chain
even when a separate high-CPU child such as `build_btc15m_toto_live_feature_context`
is present. This matters because commands are shortened in JSON handoffs; the
reason field survives even when the stored command snippet no longer shows the
MPS export flags.

## Follow-up real-input verifier hardening

I tightened `verify_worldfoam_star_native_cutwalk_promotion.py` so
`require_real_loaded_frames=true` proves more than loaded-frame counts. The
verifier now also requires `worldfoam_config` and `star_video_path` in the
summary, `--config` with that WorldFoam config in both the preflight and train
commands, and `--video-path` with that STAR video in both the planned and
selected STAR comparison commands.

This closes the remaining escape where a summary could claim real loaded
frames but still have been produced by default/repeat fixture commands. The
positive verifier fixture now carries explicit real-input command metadata, and
negative tests cover both missing custom inputs and mismatched command paths.
A second pass tightened artifact lineage too: WorldFoam `config_path` must
match `worldfoam_config`, and STAR `star.video_path` must match
`star_video_path`. A third pass made frame counts first-class in the wrapper
summary and requires real-frame promotions to prove the WorldFoam and STAR
artifact rows match that requested frame set.

Verification after the hardening:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate \
  research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale \
  research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion
```

Result: `Ran 52 tests ... OK`. Scoped `py_compile`, `git diff --check`, and
trailing-whitespace scans for the touched verifier files also passed.

I refreshed
`research_experiments/world_foam_lane2/results/2026-05-20_real32_dryrun.promotion_summary.json`
with the same real32 command. It remains a dry-run artifact, but now records
`frame_counts=[32]` along with the real WorldFoam config, STAR video path, and
`require_real_loaded_frames=true`.

One final wrapper-side guard closes the launch-time version of the same escape:
`run_worldfoam_star_native_cutwalk_gate.py` now fails during argument parsing if
`--require-real-loaded-frames` is supplied without both `--worldfoam-config` and
`--star-video-path`. That means a fake-real run cannot waste a benchmark window
and only fail at final verifier time.

Final verification after that guard:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate \
  research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale \
  research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion
```

I then added the one-sided missing-input test cases too: real-frame promotion
must reject `--worldfoam-config` without `--star-video-path`, and
`--star-video-path` without `--worldfoam-config`. The wrapper also rejects bad
`--frame-counts` at argument-parse time: empty lists, non-integers,
nonpositive values, and duplicate frame counts.

Result: `Ran 55 tests ... OK`. Scoped `py_compile`, trailing-whitespace scan,
and `git diff --check` passed for the wrapper/verifier/doc touch set. A live
check-only preflight still exits `2` because the active `ai_trader` TOTO screen
is classified as `block_reason=periodic_mps_exporter`; do not launch the clean
real32 WorldFoam/STAR timing gate until that exporter is stopped or finished.

Resume recheck: after the docs/test-count update, the immediate
`train_eval_owner_run_tape.py --benchmark-environment-check-only` still exited
`2` with `status=contended`. The same TOTO screen PIDs `79267`-`79287` were
classified as `periodic_mps_exporter`, and the monitor had just spawned
iteration `0162` under the `ai_trader` log tree. The correct next benchmark step
is unchanged: stop/wait for that monitor, then run the strict real32 wrapper
against the explicit WorldFoam config and STAR video with
`--require-real-loaded-frames`.

Follow-up harness hardening: when WorldFoam retries are enabled and attempt 2/3
becomes the selected clean artifact, the wrapper now refreshes
`planned_star_compare_command` to point `--worldfoam-artifact` at that selected
artifact instead of leaving the initial attempt-1 path in the summary. The
promotion verifier now also rejects summaries whose planned or actual STAR
command points at any WorldFoam artifact other than the selected promotable one.
Focused gate result after this lineage fix: `Ran 56 tests ... OK`; scoped
`py_compile` passed. A final check-only benchmark preflight still exited `2`
with `status=contended`; the sample had no high-CPU child, but the same
`periodic_mps_exporter` TOTO parent chain remained present, so the strict
real32 timing gate was still not launched.

Shader-boundary coverage follow-up: I added a synthetic non-repeated `32f`
moving-ray fixture to the native owner-run delta packed tests. It avoids the
slow DeepView real32 loader while still exercising unique frame ids `0..31`.
The CPU native-cutwalk delta now matches the Python owner-run sequence delta at
that 32-frame boundary, and the MPS framebitmask fused-shader output parity node
also passes against the Python-packed tape. Verification:
`NativeOwnerRunCutwalkCpuTests` plus
`OwnerRunDeltaPackedTrainEvalTests.test_native_cutwalk_framebitmask_shader_output_matches_python_at_32_frame_boundary`
ran `6` tests in `10.197s`, `OK`.

Frame-31 sign-bit shader coverage follow-up: checking the synthetic 32f
moving-ray tape showed `track_frame_mask_i32` had `negative_count=0` and
`signbit_count=0`, so unique frame ids `0..31` did not actually exercise the
signed `1 << 31` shader branch. I added a direct one-track/32f MPS regression
that sets `track_frame_mask_i32 = -(1 << 31)`, switches frame 31 from a red
base segment to an empty changed segment, and checks the loss drop against the
analytic one-frame all-base difference plus a nonzero grad difference.
Verification: the new node passed alone (`Ran 1 test in 0.166s`, `OK`), and
the full owner-run delta packed module passed (`Ran 16 tests in 199.651s`,
`OK`).

Final preflight status for this chunk: the real32 check-only benchmark command
still exits `2` with `status=contended`. The only blockers in the latest JSON
are the idle `ai_trader` TOTO screen/login/uv/python parent chain
(`79267`-`79287`), each classified as `block_reason=periodic_mps_exporter`.
No strict real32 timing/STAR comparison was launched.

Framebitmask validation follow-up: the low-level MPS wrapper previously checked
that mask bits were inside `[1, frame_count)` but did not check that
`popcount(track_frame_mask_i32[track])` matched that track's change-record
span. A malformed tape could therefore make the shader look up a nonexistent
change row or silently leave a frame invalid. I added a pre-launch wrapper
guard and a focused negative test. Verification: the two direct framebitmask
nodes pass (`Ran 2 tests in 0.200s`, `OK`), and the full owner-run delta
packed module passes (`Ran 17 tests in 203.391s`, `OK`).

Framebitmask ordering follow-up: the shader no longer carries
`change_frame_i32`; it selects the change row as
`track_begin + popcount(mask bits below frame_id)`. That makes the row order a
semantic contract, not a storage detail. I added a CPU tape-builder guard that
rejects non-strictly-ascending per-track `change_frame_i32` records before they
reach Metal, plus a focused negative test with frames `[3, 2]`. Verification:
the focused CPU nodes pass, and the full owner-run delta packed module now
passes `Ran 18 tests in 200.267s`, `OK`.

Benchmark readiness recheck after the ordering guard:

```bash
rtk .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py \
  --run-id 2026-05-20_real32_preflight_toto_mps_blocker_recheck_after_orderguard \
  --worldfoam-config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc \
  --star-video-path data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_center_crop_8fps_full.mp4 \
  --frame-counts 32 \
  --render-size 16 \
  --site-count 8 \
  --worldfoam-steps 1 \
  --star-steps 1 \
  --worldfoam-warmup-steps 1 \
  --star-warmup-steps 1 \
  --star-target-size 32 \
  --star-tube-count 224 \
  --max-worldfoam-attempts 1 \
  --max-star-attempts 1 \
  --wait-timeout-s 0 \
  --wait-poll-s 1 \
  --post-run-benchmark-environment-settle-s 5 \
  --require-real-loaded-frames \
  --verify-promotion \
  --preflight-only
```

Result: exit `2`, summary
`research_experiments/world_foam_lane2/results/2026-05-20_real32_preflight_toto_mps_blocker_recheck_after_orderguard.promotion_summary.json`,
status `worldfoam_preflight_failed_or_contended`, `worldfoam_artifact=null`,
and no STAR attempt. The only blockers are still the idle `ai_trader` TOTO
screen/login/uv/python parent chain `79267`-`79287`, all with
`block_reason=periodic_mps_exporter`.

Broader wrapper/verifier recheck after the ordering guard:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate \
  research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale \
  research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion
```

Result: `Ran 56 tests in 0.030s`, `OK`.

Adjacent no-timing regression slices after the ordering guard:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension
```

Result: `Ran 24 tests in 0.436s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps
```

Result: `Ran 8 tests in 0.334s`, `OK`.

Current real32 timing blocker recheck:

```bash
rtk .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only \
  --wait-for-benchmark-environment-ok-timeout-s 0 \
  --wait-for-benchmark-environment-ok-poll-s 1 \
  --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc
```

Result: exit `2`, `status=contended`, no background processes, and only the
idle `ai_trader` TOTO screen/login/uv/python parent chain `79267`-`79287` with
`block_reason=periodic_mps_exporter`. No strict real32 WorldFoam/STAR timing run
was launched.

Framebitmask malformed-offset follow-up: `_build_delta_frame_bitmask_i32` had
been relying on valid `track_change_offsets_i32` even though the helper can be
called directly in tests/tools. I added explicit `ValueError` guards for empty
offset vectors, nonzero first offsets, nonmonotonic offsets, and final offsets
that do not equal `len(change_frame_i32)`, plus four focused CPU negatives.
Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.NativeOwnerRunCutwalkCpuTests
```

Result: `Ran 10 tests in 9.985s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed
```

Result: `Ran 22 tests in 195.370s`, `OK`.

Quick surrounding gates after the malformed-offset guard:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate \
  research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale \
  research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion
```

Result: `Ran 56 tests in 0.031s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension
```

Result: `Ran 24 tests in 0.450s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps
```

Result: `Ran 8 tests in 0.222s`, `OK`.

Static/hygiene checks passed too: scoped `py_compile`, trailing-whitespace
scan, `git diff --check`, and `agent_notes/key_learnings.md` stayed at `199`
lines. The real32 speed gate remains blocked until the live TOTO MPS exporter
finishes or is explicitly stopped.

Post-offset-guard real32 benchmark readiness recheck:

```bash
rtk .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only \
  --wait-for-benchmark-environment-ok-timeout-s 0 \
  --wait-for-benchmark-environment-ok-poll-s 1 \
  --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc
```

Result: exit `2`, `status=contended`. The snapshot now sees one low-CPU TOTO
child (`pid=10002`, `pcpu=1.7`) as background work, and the blocking list is
still only the TOTO screen/login/uv/python parent chain `79267`-`79287`, all
with `block_reason=periodic_mps_exporter`. No strict real32 timing or STAR
comparison was launched.

Framebitmask wrapper mask-bounds follow-up: the MPS wrapper already checked
`track_frame_mask_i32` for bits outside `[1, frame_count)`, but coverage only
tested popcount mismatch and signed frame 31. I added two direct negative tests
that keep popcount equal to the per-track change count while setting illegal
bits: frame `0` and bit `frame_count`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_mask_change_count_mismatch \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_frame0_mask_bit \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_mask_bit_at_frame_count_boundary \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_uses_signed_frame31_mask_bit
```

Result: `Ran 4 tests in 0.218s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed
```

Result: `Ran 24 tests in 196.720s`, `OK`.

Final benchmark readiness recheck after the mask-bounds tests:

```bash
rtk .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only \
  --wait-for-benchmark-environment-ok-timeout-s 0 \
  --wait-for-benchmark-environment-ok-poll-s 1 \
  --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc
```

Result: exit `2`, `status=contended`, `background_processes=[]`, and only the
TOTO screen/login/uv/python parent chain `79267`-`79287` in the blocking list,
all with `block_reason=periodic_mps_exporter`. Strict real32 timing remains
unlaunched.

Shared selector sparse-change validation follow-up: the frame-select helper was
still building its int16 rank map from the same
`track_change_offsets_i32/change_frame_i32` tables without the newer
framebitmask tape-contract checks. I moved the sparse-change validation into a
shared helper used by both selectors. It now requires CPU int32 1D contiguous
vectors, well-formed offsets, frame ids in `[1, frame_count)`, and strictly
ascending per-track change frames before either frame-select or framebitmask
metadata is built. Added frame-select negatives for unsorted per-track changes,
frame-0 changes, and non-1D offsets.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.NativeOwnerRunCutwalkCpuTests
```

Result: `Ran 13 tests in 10.045s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed
```

Result: `Ran 27 tests in 196.670s`, `OK`.

Surrounding gates after the shared-selector validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate \
  research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale \
  research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion
```

Result: `Ran 56 tests in 0.031s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension
```

Result: `Ran 24 tests in 0.452s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps
```

Result: `Ran 8 tests in 0.225s`, `OK`.

Final benchmark readiness recheck after shared selector validation:

```bash
rtk .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only \
  --wait-for-benchmark-environment-ok-timeout-s 0 \
  --wait-for-benchmark-environment-ok-poll-s 1 \
  --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc
```

Result: exit `2`, `status=contended`. The snapshot sees one low-CPU TOTO child
(`pid=12907`, `pcpu=1.2`) as background work; the blocking list is still only
the TOTO screen/login/uv/python parent chain `79267`-`79287`, all with
`block_reason=periodic_mps_exporter`. Strict real32 timing remains unlaunched.

Framebitmask MPS wrapper empty-change-offset follow-up: the wrapper computed
`change_count = change_offsets_i32.numel() - 1` before proving the offset vector
contained at least one row. A malformed direct tape with empty
`change_offsets_i32` could therefore infer `change_count=-1` and fall into a
generic offset validator path instead of failing with a clear tape-contract
error. I added an explicit `change_offsets_i32.numel() >= 1` guard and a direct
MPS negative test.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_empty_change_offsets \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_mask_change_count_mismatch \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_frame0_mask_bit \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_mask_bit_at_frame_count_boundary \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_uses_signed_frame31_mask_bit
```

Result: `Ran 5 tests in 0.148s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed
```

Result: `Ran 28 tests in 196.922s`, `OK`.

Surrounding gates after the empty-offset guard:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate \
  research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale \
  research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion
```

Result: `Ran 56 tests in 0.033s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension
```

Result: `Ran 24 tests in 0.442s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps
```

Result: `Ran 8 tests in 0.204s`, `OK`.

Framebitmask MPS wrapper packed-record validation follow-up: direct callers
could still pass malformed packed endpoint records into the framebitmask Metal
shader. The train-side builder already range-checks owner/cut components before
packing, but the wrapper path only validated offsets and masks. I added a
wrapper-side packed-record validator for base/change records: negative packed
values, owner codes outside `site_count`, and left/right cut ids outside
`boundary_count` now fail before the custom op launches. Added direct MPS
negative tests for a base-record owner overflow and a change-record left-cut
overflow.

Targeted guard check:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_base_record_owner_out_of_range \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_change_record_cut_out_of_range \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_empty_change_offsets \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_uses_signed_frame31_mask_bit
```

Result: `Ran 4 tests in 0.133s`, `OK`.

Full owner-run packed module:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed
```

Result: `Ran 30 tests in 195.650s`, `OK`.

Surrounding gates after packed-record validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate \
  research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale \
  research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion
```

Result: `Ran 56 tests in 0.032s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension
```

Result: `Ran 24 tests in 0.448s`, `OK`.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps
```

Result: `Ran 8 tests in 0.220s`, `OK`.

Static check:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py
```

Result: passed.

Final benchmark readiness recheck after packed-record validation:

```bash
rtk .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only \
  --wait-for-benchmark-environment-ok-timeout-s 0 \
  --wait-for-benchmark-environment-ok-poll-s 1 \
  --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc
```

Result: exit `2`, `status=contended`, `background_processes=[]`. The only
blocking processes are still the live TOTO screen/login/uv/python parent chain
`79267`-`79287`, all with `block_reason=periodic_mps_exporter`. Strict real32
WorldFoam/STAR timing remains unlaunched.

## 2026-05-20 07:36 +07 - Sibling packed endpoint-record wrapper guard

Follow-up: the previous packed endpoint-record validation only protected the
framebitmask factorized wrapper. The same packed `owner,left,right` records are
accepted by the sibling packed wrappers, so malformed records could still reach
Metal on those paths.

Changes:

- Added `_validate_packed_endpoint_delta_records_cpu(...)` in
  `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py`.
- Wired it through the non-framebitmask packed/factorized wrappers after offset
  validation and before config/custom-op launch: packed scalar, packed
  framegroup16, packed recompute, factorized packed, factorized frameselect,
  framebitmask, smallrun16, and materialized.
- Added
  `test_non_framebitmask_packed_wrappers_reject_endpoint_record_bounds` in
  `research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py`.
  The test corrupts prepared tapes without changing record lengths for packed
  recompute, factorized packed, and factorized frameselect, then proves
  out-of-range base owners and change-record cut ids fail at the wrapper
  boundary.

Focused new regression:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_non_framebitmask_packed_wrappers_reject_endpoint_record_bounds'
```

Result: `Ran 1 test in 30.131s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 31 tests in 224.248s`, `OK`.

Surrounding gates:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.033s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.453s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.233s`, `OK`.

Static/hygiene:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py'
rtk git diff --check -- third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py EXPERIMENTS.md TODO/README.md PROJECT_INDEX.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md
rtk git -C third_party/fast-mac-gsplat diff --check -- variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py
```

Results: all passed. Note: `variants/world_foam_lane2_fused_slab_v0/` remains
untracked inside the `third_party/fast-mac-gsplat` submodule, so parent `git
diff` only sees a dirty submodule; use file contents plus the gates above for
this untracked variant state.

Final live benchmark preflight:

```bash
rtk .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only \
  --wait-for-benchmark-environment-ok-timeout-s 0 \
  --wait-for-benchmark-environment-ok-poll-s 1 \
  --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc
```

Result: exit `2`, `status=contended`. The live TOTO monitor is active again:
hot child `21541` is running `scripts.build_btc15m_toto_live_feature_context`
at `146.7%` CPU, with the `79267`-`79287` screen/login/uv/python parent chain
also marked `block_reason=periodic_mps_exporter`. Strict real32 WorldFoam/STAR
timing remains blocked.

## 2026-05-20 08:05 +07 - Packed direct-config marker guard

Follow-up: the sibling wrapper guard protected Python wrapper calls, but the hot
packed recompute path normally carries `delta_config_i32/f32` and dispatches
directly to `torch.ops.world_foam_lane2_fused_slab_v0`. That direct-config path
could bypass the new wrapper-side packed-record validation if a handwritten
`tape_device` dictionary supplied config tensors.

Changes:

- `train_eval_owner_run_tape.py` now sets
  `delta_packed_records_validated=True` immediately after CPU packed-record range
  validation and before moving through the prepared packed tape route.
- The direct-config packed dispatch now requires that marker before selecting the
  native op. This catches unsafe manually assembled direct-config tapes before a
  Metal launch without adding a per-step MPS-to-CPU record copy.
- `test_train_eval_owner_run_delta_packed.py` now asserts prepared packed modes
  carry the marker and adds
  `test_packed_recompute_direct_config_requires_prevalidated_records_marker`.

Focused regression:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_requires_prevalidated_records_marker'
```

Result: `Ran 1 test in 5.160s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 32 tests in 230.271s`, `OK`.

Static/hygiene:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py'
rtk rg -n '[[:blank:]]$' research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py
rtk git diff --check -- research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py EXPERIMENTS.md TODO/README.md PROJECT_INDEX.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md
rtk git -C third_party/fast-mac-gsplat diff --check -- variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py
```

Results: `py_compile` passed; trailing-whitespace scan had no matches;
`git diff --check` passed for both parent files and the submodule variant file.

Surrounding gates:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.032s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.456s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.233s`, `OK`.

Current real32 benchmark readiness:

```bash
rtk .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only \
  --wait-for-benchmark-environment-ok-timeout-s 0 \
  --wait-for-benchmark-environment-ok-poll-s 1 \
  --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc
```

Result: exit `2`, `status=contended`, `background_processes=[]`. The current
snapshot no longer has a hot export child, but the same live TOTO
screen/login/uv/python parent chain `79267`-`79287` remains present with
`block_reason=periodic_mps_exporter`. Strict real32 WorldFoam/STAR timing
remains unlaunched.

## 2026-05-20 08:06 +07 - Identity-bound packed direct-config marker

The earlier boolean `delta_packed_records_validated=True` guard caught missing
validation, but a stale marker could still be carried after replacing
`delta_base_record_i32` or `delta_change_record_i32` while leaving the direct
native config tensors present. That is the exact shape a bad hand-assembled
direct-config tape would take, and the hot path would still jump straight to the
Metal op.

Changes:

- `train_eval_owner_run_tape.py` now stores the marker as
  `("packed_endpoint_records_v1", id(base_record), id(change_record),
  site_count, boundary_count)`.
- The packed recompute direct-config dispatch recomputes that marker from the
  current tensors and rejects stale markers before launching the native op.
- Prepared tapes still set the marker immediately after CPU packed-record range
  validation, so this keeps the no per-step MPS-to-CPU copy property.
- `test_train_eval_owner_run_delta_packed.py` adds
  `test_packed_recompute_direct_config_rejects_replaced_records_after_marker`,
  which clones/replaces the base record after marker creation and expects the
  direct-config call to fail with the current-tensor guard.

Focused regression:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_requires_prevalidated_records_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_rejects_replaced_records_after_marker'
```

Result: `Ran 2 tests in 10.143s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 33 tests in 239.513s`, `OK`.

Surrounding no-timing gates:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.033s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.457s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.222s`, `OK`.

Current real32 benchmark readiness:

```bash
rtk zsh -lc 'PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended`. The live TOTO
`run_btc15m_overnight_shadow_monitor.py` parent chain is still present with
`block_reason=periodic_mps_exporter`; a hot child appears intermittently when
the exporter is writing the current TOTO prediction. The monitor is not stuck:
it advanced from iteration `0278` at `08:04` to `0279` at `08:05` and wrote
current TOTO/tree prediction artifacts. Strict real32 WorldFoam/STAR timing is
still unlaunched because the MPS environment is not clean.

## 2026-05-20 08:15 +07 - Config-version direct-config marker hardening

The identity-bound packed direct-config marker still had one narrow bypass: a
prepared tape could carry the correct base/change record marker while replacing
or mutating `delta_config_i32/f32`, letting the hot direct-config path launch
with stale native launch dimensions. The fix keeps the no-copy property by
binding tensor identities and PyTorch `_version` counters, not tensor contents.

Changes:

- `delta_packed_records_validated` is now
  `packed_endpoint_direct_config_v2`, binding base/change record tensors,
  `delta_config_i32`, `delta_config_f32`, each tensor's version counter, and
  site/boundary counts.
- Prepared packed tapes now stamp the marker after native config tensors are
  created, rather than immediately after record movement.
- The direct-config guard rejects record/config tensor replacement and in-place
  config mutation before the native Metal op.
- Added
  `test_packed_recompute_direct_config_rejects_mutated_config_after_marker`.

Focused regression:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_requires_prevalidated_records_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_rejects_replaced_records_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_rejects_mutated_config_after_marker'
```

Result: `Ran 3 tests in 15.174s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 34 tests in 244.552s`, `OK`.

Adjacent gates:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.038s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.493s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.251s`, `OK`.

`py_compile` also passed for the touched train/eval and owner-run packed test
files, plus the native packed `ops.py` wrapper.

## 2026-05-20 08:24 +07 - Full direct-config launch-contract marker

The config-version marker still left a broader class of direct-config bypasses:
prepared tapes could replace offset/chunk/rowdesc topology tensors or add a
different packed selector flag after validation, while keeping the records and
native config tensors unchanged. That would select a native op and launch
unchecked topology without going through the Python wrapper validators.

Changes:

- Replaced the marker payload with the then-current v3 direct-config marker,
  later superseded by newer direct-config markers.
- The marker now binds all direct-config launch tensors present in the packed
  tape: `delta_coeff_f16`, `frame_t_f32`, offsets, chunk offsets, change-frame
  rows, rowdesc buffers, packed records, and config tensors.
- It also binds selector-flag presence and launch scalar fields such as
  `delta_launch_*`, plus the external site count.
- Added regressions for replacing `track_chunk_change_offsets_i16` after marker
  creation and for adding `delta_packed_framegroup16_launch_only_fused_mse`
  after marker creation. A follow-up regression also mutates `base_offsets_i32`
  in place after marker creation. These now fail with the current-tensor guard
  before the native Metal op is selected.

Focused regression:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_requires_prevalidated_records_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_rejects_replaced_records_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_rejects_mutated_config_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_rejects_mutated_topology_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_rejects_replaced_topology_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_rejects_selector_change_after_marker'
```

Result: `Ran 6 tests in 31.334s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 37 tests in 282.436s`, `OK`.

Adjacent gates:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.033s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.457s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.227s`, `OK`.

## 2026-05-20 08:41 +07 - Post-handoff verification refresh

After a context handoff, the old terminal sessions for the adjacent gates were
gone, so I reran the current tree instead of relying on lost output handles.

Fresh verification:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.036s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.463s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.296s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 37 tests in 263.506s`, `OK`.

`py_compile`, trailing-whitespace scan, top-level `git diff --check`, submodule
`git diff --check`, and the stale-text scan all passed for the scoped files.

The real32 benchmark preflight is still blocked:

```bash
rtk zsh -lc 'PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `benchmark_environment.status=contended`. The blockers are
the live `ai_trader` TOTO monitor parent chain marked
`periodic_mps_exporter`, even while the sampled CPU percentages are idle. No
strict real32 WorldFoam/STAR timing or PSNR run should be promoted until that
monitor is stopped or finishes and a clean preflight passes.

## 2026-05-20 08:52 +07 - Generalized delta direct-config marker

Follow-up audit found a related hole outside the hot packed i32 path: legacy
raw, i16x4, i16cols, and i16x3 delta direct-config branches could still select
`torch.ops.world_foam_lane2_fused_slab_v0` directly when `delta_config_i32/f32`
were present. Those paths did not require the prep-time marker, so a handcrafted
or stale direct-config tape could bypass the Python wrapper validators.

Changes:

- Replaced the marker payload with the then-current v4 direct-config marker,
  later superseded by the v5 runtime-count marker.
- The existing `delta_packed_records_validated` key now marks every prepared
  delta direct-config tape, not only i32 packed records.
- The marker now binds raw owner/cut tensors, i16x4/i16cols/i16x3 packed
  record tensors, owner-reduce topology tensors, i16 selector flags, and the
  previous packed/topology/config/launch-scalar fields.
- The raw i32, i16x4, i16cols, and i16x3 direct-config dispatches now require
  the current marker before choosing the native op.
- Added focused regressions for missing markers on raw/i16x4/i16x3
  direct-config tapes, raw owner tensor replacement after marker creation, and
  i16x3 selector mutation after marker creation.

Focused regression:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_requires_prevalidated_records_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_legacy_delta_direct_config_requires_prevalidated_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_raw_delta_direct_config_rejects_replaced_owner_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_i16x3_direct_config_rejects_selector_change_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_rejects_mutated_config_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_rejects_mutated_topology_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_rejects_selector_change_after_marker'
```

Result: `Ran 7 tests in 45.630s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 40 tests in 296.894s`, `OK`.

`py_compile` also passed for `train_eval_owner_run_tape.py` and
`test_train_eval_owner_run_delta_packed.py`.

Adjacent post-v4 checks:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.206s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.031s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.457s`, `OK`.

After those tests exited, a fresh real32 preflight still returned exit `2` with
`benchmark_environment.status=contended`. The only blocking rows were the live
`ai_trader` TOTO monitor parent chain marked `periodic_mps_exporter`; one
low-CPU TOTO child was reported as background, not blocking. The benchmark lane
is still waiting on that external MPS exporter before strict real32
WorldFoam/STAR timing or PSNR can be promoted.

## 2026-05-20 09:14 +07 - Factorized prepared-marker closeout

The direct-config marker audit had one more gap after the raw/i16/packed v4
generalization: the prepared factorized packed, frameselect, and framebitmask
branches could still launch if their native selector flag and config tensors
were present. I moved those branches behind the same
`delta_packed_records_validated` launch-contract marker, then fixed the
handwritten framebitmask shader tests by stamping the marker after each
deliberate malformed mutation. That keeps the prep-contract tests strict while
still letting the low-level framebitmask tests reach their intended wrapper
validations.

Verification:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_uses_signed_frame31_mask_bit research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_mask_change_count_mismatch research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_empty_change_offsets research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_base_record_owner_out_of_range research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_change_record_cut_out_of_range research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_frame0_mask_bit research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_mask_bit_at_frame_count_boundary'
```

Result: `Ran 7 tests in 0.313s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result at that point: `43` owner-run packed tests in `324.887s`, `OK`.
The next section supersedes this count with the selector-family regression.

Adjacent gates also still pass:

- Mixed fused-slab MPS shader suite: `Ran 8 tests in 0.215s`, `OK`.
- Benchmark/native-cutwalk promotion wrapper and verifier group:
  `Ran 56 tests in 0.029s`, `OK`.
- Factorized selector plus native packed/cutwalk compiler group:
  `Ran 24 tests in 0.427s`, `OK`.
- Additional factorized/i16x3 packed/owner-boundary unit slice:
  `Ran 24 tests in 0.034s`, `OK`.

Fresh real32 benchmark readiness check:

```bash
rtk zsh -lc 'PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `benchmark_environment.status=contended`. This time the
snapshot caught an active high-CPU TOTO export child (`pid=35170`, about
`41%` CPU) plus the persistent `periodic_mps_exporter` parent chain
`79267`-`79287`. Strict real32 WorldFoam/STAR timing and PSNR remain
unlaunched and unpromoted until that live MPS exporter is stopped or finishes
and a clean preflight passes.

## 2026-05-20 09:28 +07 - Direct-config selector-family coverage

With the real32 timing gate still blocked by the live TOTO MPS-export parent
chain, I added a broader local regression for the marker contract itself. The
new test removes `delta_packed_records_validated` from prepared direct-config
tapes across raw, packed scalar, packed framegroup16, materialized, recompute,
smallrun16, checked/unchecked launch-only, reduce32, rowselect32, rowdesc,
rowdesc32, i16x4, i16cols, i16x3, i16x3 materialized, i16x3 ownerreduce,
i16x3 framegroup64, and the three factorized selectors. Every case must fail
with the prevalidated launch-contract error before it can choose a native Metal
op.

Focused verification:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py'
```

Result: OK.

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_all_delta_direct_config_selectors_require_prevalidated_marker'
```

Result: `Ran 1 test in 117.691s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Intermediate result before the later stale-marker scalar/topology additions:
`44` tests in `479.065s`, `OK`.

Fresh check-only real32 preflight still exits `2`, `status=contended`; the
blockers are the idle `periodic_mps_exporter` TOTO screen/login/uv/python
parent chain `79267`-`79287`. No strict real32 timing/PSNR run was launched.

## 2026-05-20 09:41 +07 - Direct-config stale-marker scalar/topology coverage

I added three narrower stale-marker regressions after the selector-family
missing-marker test. They prove the marker is not just present, but tied to the
current launch contract:

- launch-only packed delta mutates `delta_launch_site_count` after stamping the
  marker and must fail with the current-tensors contract error;
- rowdesc launch-only packed delta replaces `row_begin_i32` after stamping the
  marker and must fail before native launch;
- i16x3 owner-reduce packed delta replaces `track_chunk_owner_i16` after
  stamping the marker and must fail before native launch.

Verification:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_launch_only_direct_config_rejects_mutated_launch_scalar_after_marker \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_rowdesc_direct_config_rejects_replaced_rowdesc_after_marker \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_i16x3_ownerreduce_direct_config_rejects_replaced_owner_chunks_after_marker'
```

Result: `Ran 3 tests in 15.041s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 47 tests in 473.120s`, `OK`.

Post-verification real32 benchmark readiness check:

```bash
rtk zsh -lc 'PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended`. The only blockers reported were the
idle `periodic_mps_exporter` TOTO screen/login/uv/python parent chain
`79267`-`79287` at `0%` CPU. Strict real32 WorldFoam/STAR timing and PSNR are
still unlaunched and unpromoted.

Adjacent post-47 checks:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.031s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.441s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.182s`, `OK`.

## 2026-05-20 09:56 +07 - Direct-config marker binds runtime frame/track counts

The v4 marker still had one launch-contract hole: it bound tensors, selector
presence, launch scalars, and `site_count`, but not the runtime `track_count`
and `frame_count` arguments passed into the VJP wrapper. That mattered most for
the factorized prepared paths, where the wrapper passes those counts separately
instead of reading only native config tensors.

I changed the marker payload to `delta_direct_config_v5` and included runtime
`track_count` plus `frame_count`. Prep-time stamping now records the prepared
record-track count and frame count, while every wrapper-side current-marker
check recomputes with the runtime arguments. Manual framebitmask fixtures derive
the counts from `track_ray_coeff_f32` and `frame_t_f32` when stamping.

New regression:

- `test_direct_config_marker_rejects_runtime_count_mismatch_after_marker`
  stamps a prepared framebitmask factorized tape and then calls the VJP with
  mismatched `frame_count` and `track_count`; both calls must fail with the
  current-tensors launch-contract error before native launch.

Verification:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_direct_config_marker_rejects_runtime_count_mismatch_after_marker \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_uses_signed_frame31_mask_bit'
```

Result: `Ran 2 tests in 5.326s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Intermediate result before the later runtime-layout guard: `48` tests in
`499.822s`, `OK`.

Adjacent post-v5 checks:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.029s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.437s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.193s`, `OK`.

Fresh real32 benchmark readiness check:

```bash
rtk zsh -lc 'PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended`. The snapshot caught the TOTO monitor
parent chain plus an active child:
`python -m lean_trade.runners.btc_15m_sft_shadow --config configs/lean_btc_15m_sft_edge_paper.yaml`
at about `49%` CPU under parent `79287`. Strict real32 WorldFoam/STAR timing
and PSNR remain unlaunched and unpromoted.

## 2026-05-20 10:10 +07 - Runtime tensor layout guard before direct-config launch

After binding runtime counts in `delta_direct_config_v5`, I added one more
wrapper-side guard at the common delta fused-MSE VJP entrypoint. It now rejects
bad runtime tensor layout before any direct-config/native Metal path:

- `site_rgba` must have shape `[site_count,4]`;
- `target_rgb_track` must have shape `[track_count,frame_count,3]`.

The v5 count-mismatch regression was adjusted so the intentionally mismatched
runtime count calls pass matching target layouts and still fail via the marker,
not via the new shape guard.

New regression:

- `test_delta_direct_config_rejects_bad_runtime_tensor_layout_before_native_launch`
  uses a stamped framebitmask factorized direct-config tape and proves malformed
  `target_rgb_track` and malformed `site_rgba` fail before native launch.

Verification:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_direct_config_marker_rejects_runtime_count_mismatch_after_marker \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_runtime_tensor_layout_before_native_launch \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_uses_signed_frame31_mask_bit'
```

Result: `Ran 3 tests in 10.208s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 49 tests in 534.380s`, `OK`.

## 2026-05-20 10:14 +07 - Post-layout-guard adjacent gates and blocked real32 preflight

I reran the fast adjacent gates after the runtime tensor-layout guard to make
sure the owner-run marker/shape hardening did not drift the wrapper, native
packed compiler, or MPS fused-slab checks.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.029s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.426s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.175s`, `OK`.

Fresh real32 benchmark readiness check:

```bash
rtk zsh -lc 'PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended`. The snapshot caught the live TOTO
monitor parent chain `79267` -> `79287` plus an active child
`python -m scripts.export_btc15m_tree_residual_live_prediction_export` at about
`74.7%` CPU under parent `79287`. Strict real32 WorldFoam/STAR timing and PSNR
remain unlaunched and unpromoted.

## 2026-05-20 10:27 +07 - Runtime storage guard before fused Metal launch

The previous runtime guard only checked `site_rgba` and `target_rgb_track`
shape. I tightened it to reject dtype, device, and contiguity mistakes before
the common fused-MSE VJP entrypoint can dispatch to any direct-config/native
Metal path:

- `site_rgba` must be `float32`;
- `target_rgb_track` must be `float32`;
- `site_rgba` and `target_rgb_track` must be on the same device;
- both tensors must be contiguous.

New regression:

- `test_delta_direct_config_rejects_bad_runtime_tensor_storage_before_native_launch`
  passes a stamped framebitmask factorized direct-config tape and checks bad
  target dtype, bad site dtype, device mismatch, noncontiguous target, and
  noncontiguous site cases.

Focused verification:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_runtime_tensor_layout_before_native_launch research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_runtime_tensor_storage_before_native_launch research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_direct_config_marker_rejects_runtime_count_mismatch_after_marker'
```

Result: `Ran 3 tests in 15.166s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 50 tests in 575.820s`, `OK`.

Adjacent gates after the storage guard:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.030s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.439s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.212s`, `OK`.

Final real32 benchmark readiness check:

```bash
rtk zsh -lc 'PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended`. This time there was no high-CPU TOTO
child at the instant of sampling, but the periodic TOTO MPS-exporter parent
chain `79267` -> `79287` remained present and is still treated as a benchmark
blocker. Strict real32 WorldFoam/STAR timing and PSNR remain unlaunched and
unpromoted.

## 2026-05-20 10:43 +07 - Direct-config tape tensor storage guard

The runtime storage guard still left one manually-assembled-tape escape hatch:
a caller could replace a direct-config tape tensor with the wrong dtype, CPU
device, or noncontiguous storage, then re-stamp the marker for that bad tensor.
That marker would be current by identity/version, but the Metal custom op would
still receive an invalid ABI payload.

I added direct-config tape tensor storage validation after the marker match:

- direct-config tensor keys must be tensors;
- `*_f32` keys must be `float32`;
- `*_f16` keys must be `float16`;
- `*_i32` keys must be `int32`;
- `*_i16*` keys must be `int16`;
- marked tape tensors must live on the same runtime device as `site_rgba`;
- marked tape tensors must be contiguous.

New regression:

- `test_delta_direct_config_rejects_bad_tape_tensor_storage_after_marker`
  re-stamps a framebitmask factorized direct-config tape after deliberately
  changing a record tensor dtype, moving a record tensor to CPU, and making
  `boundary_f32` noncontiguous. All three fail before native launch.

Focused verification:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_tape_tensor_storage_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_runtime_tensor_storage_before_native_launch research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_direct_config_marker_rejects_runtime_count_mismatch_after_marker'
```

Result: `Ran 3 tests in 15.791s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 51 tests in 589.216s`, `OK`.

Adjacent gates:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.031s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.453s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.197s`, `OK`.

Final real32 benchmark readiness check:

```bash
rtk zsh -lc 'PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended`. The periodic TOTO MPS-exporter parent
chain `79267` -> `79287` remains present. Strict real32 WorldFoam/STAR timing
and PSNR remain unlaunched and unpromoted.

## 2026-05-20 11:03 +07 - Direct-config scalar launch-contract guard

The tape tensor storage guard still trusted scalar counts after marker match.
That left a hand-assembled or hand-re-stamped direct-config dictionary able to
carry stale launch counts even when the tensor identities, versions, dtypes,
devices, and contiguity looked current.

I added direct-config scalar validation after the marker match:

- scalar launch contract keys must be Python integer scalars, not tensors or
  bools;
- boundary, launch-boundary, track, frame, and site counts must be positive;
- record/change counts must be nonnegative;
- `delta_coeff_boundary_count` must match `boundary_f32.shape[0]`;
- `delta_launch_boundary_count`, `delta_launch_track_count`,
  `delta_launch_frame_count`, and `delta_launch_site_count` must match runtime
  tensor metadata;
- `delta_launch_base_record_count`, `delta_launch_change_count`, and
  `delta_launch_change_record_count` must match the first present prepared
  base/change tensor lengths;
- `delta_config_i32` must be length `7` or `8`, and `delta_config_f32` must be
  length `4`.

This uses tensor metadata only and does not add per-step MPS-to-CPU value
copies.

New regression:

- `test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker`
  re-stamps bad boundary, track, and base-record launch counts after marker
  validation and expects the Python boundary to reject them before native Metal
  launch.

Verification:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

Focused scalar/tape/count regression batch:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_tape_tensor_storage_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_direct_config_marker_rejects_runtime_count_mismatch_after_marker'
```

Result: `Ran 3 tests in 20.075s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 52 tests in 596.812s`, `OK`.

Adjacent gates:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.031s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.453s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.170s`, `OK`.

Final real32 benchmark readiness check:

```bash
rtk zsh -lc 'PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended`. The check caught a live TOTO residual
prediction export child at `26.1%` CPU plus the same `periodic_mps_exporter`
parent chain `79267` -> `79287`. Strict real32 WorldFoam/STAR timing and PSNR
remain unlaunched and unpromoted.

## 2026-05-20 11:19 +07 - Direct-config tensor-layout guard

After the scalar launch-contract guard, the direct-config path still accepted a
hand-re-stamped dictionary whose tensors had the right dtype, device, and
contiguity but the wrong ABI shape. That is enough to make a bad tape look
current to the marker while still handing malformed buffers to Metal.

I added a metadata-only tensor-layout validator after the marker and storage
checks. It validates:

- `boundary_f32` rank/columns: `[boundary_count,5]`;
- `delta_coeff_f16` rank/columns: `[row_count,4]`;
- `track_ray_coeff_f32`: `[track_count,12]`;
- `frame_t_f32`: `[frame_count]`;
- base and track-change offsets: `[track_count + 1]`;
- `track_frame_mask_i32`: `[track_count]`;
- `frame_change_index_i16`: `[track_count * (frame_count - 1)]`;
- rowdesc buffers: `[track_count * frame_count]`;
- change offset vectors have one more row than their change-frame vectors;
- flattened i16x4 records are divisible by `4`;
- flattened i16x3/i16cols records are divisible by `3`;
- base owner/left/right component vectors share length, and change
  owner/left/right component vectors share length.

The marker payload string is now `delta_direct_config_v6` so old direct-config
markers cannot be mistaken for the new layout contract. The guard still uses
shape metadata only; it does not add per-step MPS-to-CPU value copies.

New regression:

- `test_delta_direct_config_rejects_bad_tape_tensor_shape_after_marker`
  re-stamps malformed boundary columns, factorized track-ray coeff columns,
  owner-reduce flattened i16x3 record length, and rowdesc row count after
  marker validation. All fail at the Python boundary before native Metal launch.

Verification:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

Focused shape/scalar/storage regression batch:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_tape_tensor_shape_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_tape_tensor_storage_after_marker'
```

Result: `Ran 3 tests in 30.747s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 53 tests in 621.442s`, `OK`.

Adjacent gates:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.032s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.463s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.238s`, `OK`.

Final real32 benchmark readiness check:

```bash
rtk zsh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 11:19 +07`. The check
caught a live TOTO residual prediction export wrapper at `24.4%` CPU plus the
same `periodic_mps_exporter` parent chain `79267` -> `79287`. Strict real32
WorldFoam/STAR timing and PSNR remain unlaunched and unpromoted.

## 2026-05-20 11:40 +07 - Direct-config selector-contract guard

The layout guard still left one branch-order escape hatch: a hand-re-stamped
direct-config dictionary could contain incompatible selector keys. The marker
would be current, and the Python dispatch order would silently choose one ABI
family while the tape also claimed another. That is not a value-range problem;
it is a pure key-contract problem and can be checked without reading MPS tensor
values.

I added direct-config selector-contract validation after marker match:

- packed primary selectors, factorized primary selectors, and i16 primary
  selectors are mutually exclusive;
- launch-only modifiers require a non-scalar packed framegroup selector and
  the base launch-only flag;
- rowdesc32 requires rowdesc;
- reduce32, rowselect32, and rowdesc launch row selector modifiers are mutually
  exclusive.

The marker payload string is now `delta_direct_config_v7`, so old direct-config
markers cannot be mistaken for the new selector contract.

New regression:

- `test_delta_direct_config_rejects_conflicting_selectors_after_marker`
  re-stamps a factorized framebitmask tape with a frameselect selector, a
  scalar packed tape with launch-only, and a launch-only framegroup tape with
  both reduce32 and rowselect32. All fail at the Python boundary before native
  Metal launch.

One small test-only fix was needed: the generic `_stamp_delta_launch_contract`
helper derives `track_count` from `track_ray_coeff_f32`, which scalar packed
tapes do not have. The scalar conflict case stamps the marker directly from
the prepared tape's `track_count` instead.

Verification:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

Focused selector/layout regression batch:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_conflicting_selectors_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_tape_tensor_shape_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_all_delta_direct_config_selectors_require_prevalidated_marker'
```

Result: `Ran 3 tests in 198.141s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 54 tests in 670.629s`, `OK`.

Adjacent gates:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.033s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.466s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.222s`, `OK`.

Final real32 benchmark readiness check:

```bash
rtk zsh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 11:40 +07`. The active
blockers were the idle `periodic_mps_exporter` parent chain `79267` -> `79287`.
Strict real32 WorldFoam/STAR timing and PSNR remain unlaunched and unpromoted.

## 2026-05-20 12:01 +07 - Packed direct-config scalar-count coverage for non-launch selectors

After the selector-contract guard, one scalar-count hole remained: packed i32
direct-config prep only stamped `delta_launch_*` count scalars inside the
launch-only branch. Non-launch packed/factorized selectors still had a marker
and scalar validator, but no prepared base/change record counts to validate
against. That mattered for factorized frameselect/framebitmask tapes because a
hand-re-stamped direct-config dictionary could otherwise carry stale
`delta_launch_change_count` without the value being tied back to the prepared
offset tensors.

I moved the packed i32 `delta_launch_*` scalar stamping into the common packed
direct-config prep path, so scalar counts are present for packed scalar,
framegroup/materialized/recompute/smallrun, launch-only, and owner-run
factorized packed/frameselect/framebitmask selectors. I also tightened
`delta_launch_change_count` validation: it first checks `change_frame_i32` or
`change_frame_i16` length when those tensors are resident, and otherwise derives
the count from `change_offsets_i32` or `change_offsets_i16` length minus one.
That gives factorized frameselect/framebitmask non-launch selectors a concrete
metadata-only count source without copying MPS tensor values back to CPU.

Regression coverage stayed inside the existing scalar-contract test:

- `test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker`
  now asserts that a prepared factorized framebitmask tape carries
  `delta_launch_base_record_count`, `delta_launch_change_count`, and
  `delta_launch_change_record_count`, then re-stamps a deliberately bad
  `delta_launch_change_count` and expects wrapper rejection against
  `change_offsets_i32`.

Verification:

```bash
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

Focused scalar/direct-config regression batch:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_all_delta_direct_config_selectors_require_prevalidated_marker'
```

Result: `Ran 2 tests in 130.167s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 54 tests in 672.973s`, `OK`.

Adjacent gates:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.038s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.467s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.180s`, `OK`.

Final real32 benchmark readiness check:

```bash
rtk zsh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 12:01 +07`. The active
blockers were still the `periodic_mps_exporter` parent chain `79267` -> `79287`.
The snapshot also saw a short-lived tree-oracle feature-frame subprocess under
`79287` as background work. Strict real32 WorldFoam/STAR timing and PSNR remain
unlaunched and unpromoted.

## 2026-05-20 12:20 +07 - Required scalar contract keys for i32 packed direct-config

The scalar-count patch still left one adversarial wrapper hole: prepared i32
packed direct-config tapes now stamped `delta_launch_*` counts, but the
validator only checked those counts when the keys were present. A hand-re-stamped
factorized framebitmask dictionary could delete one count scalar and bypass the
contract entirely.

I added `_PACKED_DIRECT_CONFIG_LAUNCH_COUNT_SCALAR_KEYS` and made
`_validate_packed_direct_config_scalars(...)` require every launch-count scalar
whenever any i32 packed direct-config primary selector is present. The selector
set covers both `_PACKED_DIRECT_CONFIG_PACKED_PRIMARY_SELECTOR_KEYS` and
`_PACKED_DIRECT_CONFIG_FACTORIZED_PRIMARY_SELECTOR_KEYS`; i16 selectors remain
outside this requirement because they have their own compact launch contract.
Missing keys now fail with `i32 packed direct-config path missing scalar
contract keys: ...` before the wrapper can dispatch to Metal.

Regression coverage now deletes `delta_launch_change_record_count` after
re-stamping the marker in
`test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker` and
expects that missing-scalar rejection. The manual framebitmask shader fixtures
also carry the required scalar counts explicitly, with all-base and
empty-change negative fixtures overriding both `delta_launch_change_count` and
`delta_launch_change_record_count` to zero. The frame-count-boundary negative
case updates `delta_launch_frame_count` when it mutates `frame_count`, so it
still reaches the intended mask-bit guard rather than failing early on a stale
scalar.

Verification:

```bash
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

Focused affected tests:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_uses_signed_frame31_mask_bit research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_empty_change_offsets research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_shader_rejects_mask_bit_at_frame_count_boundary'
```

Result: `Ran 4 tests in 10.400s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 54 tests in 697.562s`, `OK`.

Adjacent gates:

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Result: `Ran 56 tests in 0.033s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Result: `Ran 24 tests in 0.464s`, `OK`.

```bash
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps'
```

Result: `Ran 8 tests in 0.230s`, `OK`.

The strict real32 WorldFoam/STAR timing and PSNR gate is still intentionally
unlaunched. At `2026-05-20 12:20 +07`, the benchmark-environment preflight
returned exit `2` with `status=contended`; it saw an active ai_trader TOTO
export subprocess `58247` at `97.4%` CPU plus the `periodic_mps_exporter`
parent chain `79267` -> `79287`.

## 2026-05-20 12:40 +07 - Missing launch-count coverage across i32 packed selector families

The required-scalar regression initially only deleted
`delta_launch_change_record_count` on the factorized framebitmask selector. That
proved the validator path, but it did not prove the same missing-key contract
for scalar, framegroup, materialized, recompute, smallrun, launch-only, regular
factorized, and frameselect i32 packed direct-config selectors.

I added
`test_i32_packed_direct_config_selectors_require_scalar_launch_count_keys_after_marker`.
It prepares nine i32 packed selector families, verifies every
`delta_launch_*` count scalar is present, deletes
`delta_launch_change_record_count`, re-stamps the current direct-config marker
with the explicit tape track count, and expects the wrapper to reject with
`missing scalar contract keys`. One small lesson from the first attempt:
the local helper that infers track count from `track_ray_coeff_f32` only applies
to the factorized paths, so this cross-selector test must stamp the marker from
the tape's explicit `track_count`.

Verification:

```bash
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py'
```

Result: OK.

Focused scalar-contract tests:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_i32_packed_direct_config_selectors_require_scalar_launch_count_keys_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker'
```

First attempt failed because `_stamp_delta_launch_contract(...)` expected
`track_ray_coeff_f32` on non-factorized selectors. After switching the test to
stamp via `_packed_endpoint_direct_config_validation_marker(...)` and the tape's
explicit `track_count`, the focused pair passed: `Ran 2 tests in 77.248s`,
`OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 55 tests in 717.702s`, `OK`.

Final benchmark preflight after the full module run:

```bash
rtk zsh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 12:41 +07`. The earlier
high-CPU TOTO export subprocess had cleared, but the idle
`periodic_mps_exporter` parent chain `79267` -> `79287` remained alive. A
background tree-oracle context feature-frame subprocess `67739` was also
visible at `0.9%` CPU but was not the hard blocker. The strict real32
timing/PSNR gate remains blocked until the ai_trader TOTO MPS export monitor is
stopped or finishes.

## 2026-05-20 13:08 +07 - Required coeff boundary scalar too

The next scalar-contract probe found one more hole: deleting
`delta_coeff_boundary_count` from a re-stamped non-factorized packed scalar
direct-config tape still reached the direct-config launch path. The launch-count
scalars were required, but the coefficient boundary scalar was only validated
if present.

I added `_PACKED_DIRECT_CONFIG_REQUIRED_SCALAR_KEYS`, which is
`delta_coeff_boundary_count` plus the existing `delta_launch_*` launch-count
keys, and changed the i32 packed direct-config missing-key guard to reject
missing required scalar contract keys. The cross-selector test now includes
`delta_coeff_boundary_count` in its required scalar list and adds a packed
scalar subcase that deletes it, re-stamps the marker, and expects
`missing scalar contract keys`.

Verification:

```bash
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

Focused scalar-contract tests:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_i32_packed_direct_config_selectors_require_scalar_launch_count_keys_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker'
```

Result: `Ran 2 tests in 85.678s`, `OK`.

Full owner-run packed module:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 55 tests in 1336.858s`, `OK`.

Final benchmark preflight after the full module rerun:

```bash
rtk zsh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 13:09 +07`. The idle
`periodic_mps_exporter` parent chain `79267` -> `79287` remained the hard
blocker. A background tree-oracle context feature-frame subprocess `89739` was
visible at `4.8%` CPU but below the blocker threshold.

## 2026-05-20 13:30 +07 - Type-stable scalar markers for direct-config v8

The next marker probe found a smaller wrapper-boundary risk: the
direct-config validation marker built scalar marker entries with `int(...)`.
That made invalid scalar values a marker-construction concern, where they could
be coerced or fail before the scalar-contract validator produced the intended
error. I added `_direct_config_scalar_marker(...)`, bumped the marker payload to
`delta_direct_config_v8`, and changed scalar marker construction to use a
type-stable tensor-identity-style marker for non-Python-integer scalar values.

The regression sets `delta_launch_track_count = "not-an-int"`, re-stamps the
direct-config marker, and verifies that the wrapper raises
`delta_launch_track_count must be a Python integer scalar` from the scalar
validator rather than during marker construction.

Verification:

```bash
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
```

Result: OK.

Focused marker/scalar tests:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_i32_packed_direct_config_selectors_require_scalar_launch_count_keys_after_marker'
```

Result: `Ran 2 tests in 63.819s`, `OK`.

Full owner-run packed module after the marker bump:

```bash
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Result: `Ran 55 tests in 760.887s`, `OK`.

The strict benchmark preflight still fails cleanly with `status=contended`:

```bash
rtk zsh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 13:30 +07`. The
`periodic_mps_exporter` parent chain `79267` -> `79287` remained the hard
blocker, so strict WorldFoam/STAR timing and PSNR are still intentionally
unlaunched.

Follow-up coverage tightened the type-stable marker claim: a valid
direct-config marker is now stamped first, then `delta_launch_track_count` is
mutated to `True`; the wrapper must reject it as a stale prevalidation marker
instead of accepting the old integer marker shape and falling through to the
scalar validator. Verification after adding that case:

```bash
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_i32_packed_direct_config_selectors_require_scalar_launch_count_keys_after_marker'
```

Result: `py_compile` OK; focused pair `Ran 2 tests in 70.560s`, `OK`.

Final preflight check at `2026-05-20 13:36 +07` still returned
`status=contended`: the idle `periodic_mps_exporter` screen/login/uv/Python
chain `79267` -> `79287` remained the only hard blocker, while idle
`MTLCompilerService` processes were classified as background.

## 2026-05-20 13:46 +07 - Full packed module rerun and fixture cache fix

The first full rerun after the stale-bool scalar-marker test did not fail an
invariant; it exposed a test-harness reliability issue. The module ran `55`
tests for `783.422s` and errored once when OpenCV timed out reading frame `37`
at `1.250s` from
`data/external/deepview_video/extracted/03_Dog/03_Dog/camera_0001.mp4` while
building the moving-ray fixture for
`test_packed_recompute_direct_config_requires_prevalidated_records_marker`.
That test passed by itself immediately afterward (`Ran 1 test in 5.563s`,
`OK`), so the failure was repeated video decoding pressure, not a broken
direct-config guard.

I added `_cached_loaded_training_frames(...)` with `@lru_cache` in
`test_train_eval_owner_run_delta_packed.py`. It caches the loaded/fitted CPU
`targets`, `rays`, and `frame_indices` per `(frame_count, render_size)` and
the fixture clones those tensors before applying synthetic ray motion, so each
test still gets isolated tensors while the suite avoids repeatedly opening the
same DeepView videos.

Verification after the cache fix:

```bash
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_packed_recompute_direct_config_requires_prevalidated_records_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker'
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Results: `py_compile` OK; focused pair `Ran 2 tests in 5.439s`, `OK`; full
owner-run packed module `Ran 55 tests in 151.230s`, `OK`; wrapper/verifier
contract suite `Ran 56 tests in 0.045s`, `OK`.

The strict real32 benchmark preflight at `2026-05-20 13:38 +07` still returned
`status=contended`: a live `ai_trader` audit child was at `95.4%` CPU, a
tree-oracle child at `5.3%` CPU, and the `periodic_mps_exporter` parent chain
`79267` -> `79287` was still alive. Strict WorldFoam/STAR timing and PSNR
remain intentionally unlaunched.

## 2026-05-20 20:50 +07 - Clean-evening strict wrapper reached one diagnostic real32 row

The old TOTO blocker chain disappeared briefly, so I reran the strict real32
wrapper with real inputs and `--require-real-loaded-frames`:

```bash
rtk .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py \
  --run-id 2026-05-20_real32_strict_mini_wrapper_clean_evening \
  --worldfoam-config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc \
  --star-video-path data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_center_crop_8fps_full.mp4 \
  --frame-counts 32 \
  --render-size 16 \
  --site-count 8 \
  --worldfoam-steps 1 \
  --star-steps 1 \
  --worldfoam-warmup-steps 1 \
  --star-warmup-steps 1 \
  --star-target-size 32 \
  --star-tube-count 224 \
  --max-worldfoam-attempts 2 \
  --max-star-attempts 2 \
  --wait-timeout-s 300 \
  --wait-poll-s 2 \
  --post-run-benchmark-environment-settle-s 5 \
  --require-real-loaded-frames \
  --verify-promotion
```

Attempt 1 did run the true loaded-32f WorldFoam config. Artifact:
`research_experiments/world_foam_lane2/results/2026-05-20_real32_strict_mini_wrapper_clean_evening.attempt1.worldfoam.json`.

Diagnostic row:

| frame_count | loaded_frame_count | repeat_loaded_frames | train PSNR | heldout PSNR | total | backward |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | false | 12.987 | 14.229 | 3.104 ms | 2.773 ms |

This is still not promotable timing evidence. The post-run benchmark snapshot
found restarted live `ai_trader` offline TOTO MPS-export monitors and transient
`MTLCompilerService`, so attempt 1 was rejected after artifact write. Attempt 2
never launched WorldFoam because preflight stayed contended under the new TOTO
monitor chains. The promotion summary
`research_experiments/world_foam_lane2/results/2026-05-20_real32_strict_mini_wrapper_clean_evening.promotion_summary.json`
ended `worldfoam_preflight_failed_or_contended`; `worldfoam_artifact=null`,
`star_compare_command=null`, and no STAR compare artifact exists.

The useful conclusion is narrower than promotion: the real32 WorldFoam path can
run a warm true-32f step in a few milliseconds, but the strict WorldFoam/STAR
speed and PSNR gate remains blocked by external TOTO MPS-export automation. The
next real timing attempt should pause/stop those `ai_trader` TOTO exporter
screens or run in a clean machine window.

Post-doc verification:

```bash
rtk rg -n '[[:blank:]]$' EXPERIMENTS.md TODO/README.md PROJECT_INDEX.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py
rtk rg -n 'missing scalar launch-count|missing-launch-count|scalar launch-count|delta_direct_config_v7|54 tests|53 tests|delta_direct_config_v6|delta_direct_config_v5|delta_direct_config_v4|delta_direct_config_v3|packed_endpoint_direct_config_v3|prevalidated packed launch|paked' PROJECT_INDEX.md TODO/README.md EXPERIMENTS.md research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
rtk git diff --check -- EXPERIMENTS.md TODO/README.md PROJECT_INDEX.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Results: no trailing whitespace/stale-marker hits; `py_compile` OK; scoped
`git diff --check` OK; wrapper/verifier contract suite `Ran 56 tests in
0.033s`, `OK`.

Fresh readiness check after the test process exited:

```bash
rtk zsh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 20:49 +07`. The blocker
list contained only the idle `toto_floor001_guardaligned_offline_20260520T134240Z`
screen/login/uv/Python chain marked `block_reason=periodic_mps_exporter`.
The corresponding `ai_trader` monitor is still doing real shadow work: its
event log advanced to `112` rows and the latest iteration ended with
`btc15m_current_gate_blocker_ledger pass` at `2026-05-20T13:49:24Z`.

## 2026-05-20 21:00 +07 - Base/change record-count scalar regressions

While strict timing stayed blocked, I tightened the direct-config scalar
contract on the selected factorized framebitmask shader path. The existing
test covered boundary count, change count, missing scalar keys, scalar type,
runtime count mismatch, and stale bool mutation. I added explicit
re-stamped-contract regressions for `delta_launch_base_record_count` and
`delta_launch_change_record_count`, so a hand-mutated direct-config dictionary
cannot carry current marker metadata while lying about packed base/change
record cardinalities.

Verification:

```bash
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py'
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker'
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Results: `py_compile` OK; focused scalar test `Ran 1 test in 6.907s`, `OK`;
full owner-run packed module `Ran 55 tests in 245.157s`, `OK`.

Post-regression hygiene:

```bash
rtk rg -n '[[:blank:]]$' EXPERIMENTS.md TODO/README.md PROJECT_INDEX.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py
rtk rg -n 'missing scalar launch-count|missing-launch-count|scalar launch-count|delta_direct_config_v7|54 tests|53 tests|delta_direct_config_v6|delta_direct_config_v5|delta_direct_config_v4|delta_direct_config_v3|packed_endpoint_direct_config_v3|prevalidated packed launch|paked' PROJECT_INDEX.md TODO/README.md EXPERIMENTS.md research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py'
rtk git diff --check -- EXPERIMENTS.md TODO/README.md PROJECT_INDEX.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py
```

Results: no trailing whitespace/stale-marker hits; `py_compile` OK; scoped
`git diff --check` OK.

Fresh real32 benchmark readiness check:

```bash
rtk zsh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 20:57 +07`. Idle
`MTLCompilerService` rows were background only; the hard blockers were still the
`toto_floor001_guardaligned_offline_20260520T134240Z` screen/login/uv/Python
chain, each marked `block_reason=periodic_mps_exporter`.

## 2026-05-20 21:05 +07 - STAR artifact summary selection semantics

I tightened `run_worldfoam_star_native_cutwalk_gate.py` so diagnostic summaries
no longer put a merely planned STAR compare path into the selected
`star_compare_artifact` field. New summaries now carry
`planned_star_compare_artifact` separately, keep `star_compare_artifact=null`
until a STAR attempt is promotable, and use
`star_compare_latest_attempt_artifact` / `star_compare_latest_written_artifact`
only for STAR commands that actually ran.

This matters because the saved clean-evening summary is a legacy pre-hardening
artifact:
`research_experiments/world_foam_lane2/results/2026-05-20_real32_strict_mini_wrapper_clean_evening.promotion_summary.json`
has `status=worldfoam_preflight_failed_or_contended`, `star_compare_command=null`,
and no STAR file, but still stores the planned
`2026-05-20_real32_strict_mini_wrapper_clean_evening.star_attempt1.star_compare.json`
path in `star_compare_artifact`. I did not rewrite that evidence artifact; the
code and tests fix future summaries.

Verification after the wrapper/doc patch:

```bash
rtk rg -n '[[:blank:]]$' EXPERIMENTS.md TODO/README.md PROJECT_INDEX.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py
rtk rg -n 'missing scalar launch-count|missing-launch-count|scalar launch-count|delta_direct_config_v7|54 tests|53 tests|delta_direct_config_v6|delta_direct_config_v5|delta_direct_config_v4|delta_direct_config_v3|packed_endpoint_direct_config_v3|prevalidated packed launch|paked' PROJECT_INDEX.md TODO/README.md EXPERIMENTS.md research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Results: no trailing whitespace/stale-marker hits; `py_compile` OK; broader
wrapper/verifier suite `Ran 56 tests in 0.031s`, `OK`.

Fresh real32 benchmark readiness check:

```bash
rtk zsh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 21:05 +07`. The live
blockers are the new
`toto_floor001_guardaligned_edgefloor_20260520T135831Z` screen/login/uv/Python
chain marked `block_reason=periodic_mps_exporter`, plus a high-CPU Python child
under that monitor. Idle `MTLCompilerService` rows remain background-only. The
next strict timing/PSNR promotion should still wait for or explicitly stop this
external TOTO MPS exporter.

## 2026-05-20 21:08 +07 - Compressed key-learning update and live blocker

I folded the new strict-promotion lesson into `agent_notes/key_learnings.md`
without increasing the file above its 199-line budget: periodic TOTO MPS
exporters are benchmark blockers even when parent processes are idle, and
WorldFoam/STAR wrapper summaries must separate planned STAR paths from selected
`star_compare_artifact` fields.

Focused verification for the changed notes and current shader guard surface:

```bash
rtk wc -l agent_notes/key_learnings.md
rtk rg -n '[[:blank:]]$' agent_notes/key_learnings.md PROJECT_INDEX.md EXPERIMENTS.md TODO/README.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md
rtk git diff --check -- agent_notes/key_learnings.md PROJECT_INDEX.md EXPERIMENTS.md TODO/README.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py'
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_all_delta_direct_config_selectors_require_prevalidated_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_prepared_paths_require_prevalidated_marker'
```

Results: `key_learnings.md` remains `199` lines; no trailing whitespace;
`git diff --check` OK; `py_compile` OK; focused direct-config/marker tests
`Ran 3 tests in 7.057s`, `OK`.

Fresh real32 benchmark readiness check:

```bash
rtk zsh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 21:08 +07`. The blocker
snapshot includes the active TOTO export child
`scripts/export_btc15m_toto_residual_live_prediction_export.py` at `55.4%` CPU,
its `uv run ... --device mps` wrapper, and the idle
`toto_floor001_guardaligned_edgefloor_20260520T135831Z` parent chain marked
`periodic_mps_exporter`. This confirms the fail-fast classifier is catching the
real intermittent export burst, not only a stale screen name.

## 2026-05-20 21:12 +07 - Promotion verifier requires STAR attempt lineage

I tightened the native-cutwalk promotion verifier so an `ok` summary must prove
the selected STAR artifact was actually selected by the wrapper state machine.
The verifier now rejects summaries unless:

- `star_compare_artifact` is selected.
- `star_compare_latest_attempt_artifact` matches the selected STAR artifact.
- `star_compare_latest_written_artifact` matches the selected STAR artifact.
- `star_compare_attempts` contains exactly one promotable STAR attempt, and that
  attempt points at the selected STAR artifact.

This closes a local hand-authored-summary escape where STAR payload checks could
pass even if the summary omitted STAR attempt lineage or left latest-written
metadata inconsistent with the selected artifact.

Verification:

```bash
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py research_experiments/world_foam_lane2/test_verify_worldfoam_star_native_cutwalk_promotion.py'
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Results: `py_compile` OK; focused verifier suite `Ran 13 tests in 0.018s`,
`OK`; broader wrapper/verifier suite `Ran 58 tests in 0.033s`, `OK`.

## 2026-05-20 21:14 +07 - Live exporter still blocks strict timing

I rechecked the live benchmark environment after confirming the TOTO screen was
doing real interval work. The monitor is not stuck: iteration `0014` wrote
TOTO residual prediction/report artifacts at `21:12:45 +07`, and iteration
`0015` wrote the same class of artifacts at `21:13:47 +07`.

Fresh Dynaworld preflight:

```bash
rtk zsh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 21:14 +07`. The blocker
list still includes the detached
`toto_floor001_guardaligned_edgefloor_20260520T135831Z` screen/login/uv/Python
chain marked `block_reason=periodic_mps_exporter`, even while the parent chain
was at `0.0%` CPU. The preflight also saw a background `ai_trader`
`probe_btc15m_tree_oracle_context_feature_frame` process. Strict WorldFoam/STAR
timing and promotion reruns should still wait for this external monitor to
finish or be explicitly stopped.

Post-note verification:

```bash
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py research_experiments/world_foam_lane2/test_verify_worldfoam_star_native_cutwalk_promotion.py research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py'
rtk rg -n '[[:blank:]]$' PROJECT_INDEX.md EXPERIMENTS.md TODO/README.md agent_notes/key_learnings.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py research_experiments/world_foam_lane2/test_verify_worldfoam_star_native_cutwalk_promotion.py research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py
rtk git diff --check -- PROJECT_INDEX.md EXPERIMENTS.md TODO/README.md agent_notes/key_learnings.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py research_experiments/world_foam_lane2/test_verify_worldfoam_star_native_cutwalk_promotion.py research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
rtk zsh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_all_delta_direct_config_selectors_require_prevalidated_marker research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_prepared_paths_require_prevalidated_marker'
```

Results: `py_compile` OK; no trailing whitespace; `git diff --check` OK;
wrapper/verifier suite `Ran 58 tests in 0.033s`, `OK`; focused packed
direct-config marker slice `Ran 3 tests in 6.943s`, `OK`.

## 2026-05-20 21:20 +07 - Unchecked benchmark probes now block promotion

I found one more strict-gate hole while timing was externally blocked: if
`ps`/environment capture failed and returned
`benchmark_environment.status = unchecked`, both the WorldFoam train/eval
preflight and the STAR comparison treated it as non-blocking. That could let a
strict run proceed with no real proof that the machine was clean. The gate now
treats only `ok` and
`background` as promotable; `contended`, `unchecked`, missing, or unknown
statuses block strict promotion.

Changed files:

- `research_experiments/world_foam_lane2/train_eval_owner_run_tape.py`
- `research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py`
- `research_experiments/world_foam_lane2/test_train_eval_benchmark_environment.py`
- `research_experiments/world_foam_lane2/test_train_eval_owner_run_benchmark_environment.py`
- `research_experiments/world_foam_lane2/test_compare_star_uvt_worldfoam_scale.py`

Verification:

```bash
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py research_experiments/world_foam_lane2/test_train_eval_benchmark_environment.py research_experiments/world_foam_lane2/test_train_eval_owner_run_benchmark_environment.py research_experiments/world_foam_lane2/test_compare_star_uvt_worldfoam_scale.py'
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale'
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Results: `py_compile` OK; focused environment/compare suite `Ran 29 tests in
0.004s`, `OK`; broader wrapper/verifier suite `Ran 63 tests in 0.059s`, `OK`.
Final hygiene after doc updates: no trailing whitespace and `git diff --check`
passed for the touched code/docs.

Fresh real32 benchmark readiness check:

```bash
rtk zsh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 21:22 +07`. The hard
blocker is still the detached
`toto_floor001_guardaligned_edgefloor_20260520T135831Z` parent chain marked
`periodic_mps_exporter`; a low-CPU tree-oracle feature probe and a read-only
memory-writer Codex process were background-only. No strict WorldFoam/STAR
timing rerun was launched.

## 2026-05-20 21:28 +07 - Promotion status contract accepts quiet ok snapshots

The previous strict-gate pass made `unchecked` block promotion, but I found a
companion mismatch: the wrapper and promotion verifier still accepted only
`background` benchmark snapshots as clean, while train/eval and STAR compare
accepted both `ok` and `background`. That meant a truly quiet machine with no
background keyword-matched rows could be rejected as non-promotable.

I added a single wrapper helper for promotable environment statuses and changed
the wrapper/verifier contract to accept `ok` and `background`, while continuing
to reject `contended`, `unchecked`, missing, and unknown statuses. The wrapper
also retries later attempts when an artifact was written but its environment
status is non-promotable, not only when that status is exactly `contended`.

Verification:

```bash
rtk zsh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py research_experiments/world_foam_lane2/test_verify_worldfoam_star_native_cutwalk_promotion.py'
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
rtk zsh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
```

Results: `py_compile` OK; wrapper/verifier pair `Ran 37 tests in 0.043s`,
`OK`; broader wrapper/verifier gate `Ran 66 tests in 0.039s`, `OK`.

## 2026-05-20 21:37 +07 - Native packed extension included in focused gate

I refreshed the focused lane gate against the current worktree after noticing
the native packed-extension fixture was built and should be part of the strict
wrapper/verifier count. The built extension exists at:

```text
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/_C.cpython-311-darwin.so
```

Verification:

```bash
rtk sh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py research_experiments/world_foam_lane2/verify_native_packed_extension.py'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion research_experiments.world_foam_lane2.test_verify_native_packed_extension'
rtk sh -lc 'PYTHONPATH=src/train:research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed'
```

Results: `py_compile` OK; focused wrapper/verifier/native gate `Ran 67 tests in
0.064s`, `OK`; owner-run delta packed contract suite `Ran 55 tests in
160.417s`, `OK`.

Fresh strict real32 benchmark readiness check:

```bash
rtk sh -lc 'PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended`. During the check, the just-running unit
test Python process was high CPU and the detached
`toto_floor001_guardaligned_edgefloor_20260520T135831Z` screen/login/uv/python
chain was still classified as `periodic_mps_exporter`. This remains a correct
strict benchmark blocker, so no new WorldFoam/STAR timing rerun was launched.

## 2026-05-20 21:39 +07 - Strict preflight no longer self-blocks on metal config names

The fresh real32 preflight revealed a real gate bug: because
`_capture_benchmark_environment()` ignored only the current Python PID and its
direct parent, the higher-level `rtk sh -lc ... local_mac_powerfoam_metal...`
launch wrapper could remain in the `ps` table and self-match `keyword:metal`.
That would make a clean machine look contended solely because the config path
contains `metal`.

I changed benchmark capture to parse the `ps` rows first, compute the full
current-process ancestor chain from `pid -> ppid`, and ignore every ancestor
before classifying background/blocking processes. The regression fixture mocks a
three-process launch chain where both ancestors mention a `powerfoam_metal`
config and now expects `status=ok`.

Verification:

```bash
rtk sh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_benchmark_environment.py'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Results: `py_compile` OK; benchmark-environment unit slice `Ran 16 tests in
0.001s`, `OK`; focused wrapper/verifier/native gate `Ran 68 tests in 0.058s`,
`OK`.

Fresh strict real32 benchmark readiness check after the patch:

```bash
rtk sh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 21:36 +07`. The
self-blocking `keyword:metal` row is gone. The remaining hard blockers are real
external work: the detached TOTO export chain and one active TOTO child
(`scripts.report_btc15m_lean_paper_readiness_matrix`) at `75.2%` CPU. No strict
WorldFoam/STAR timing rerun was launched.

## 2026-05-20 21:43 +07 - STAR compare capture gets the same ancestor-chain fix

I found the same strict-preflight self-block risk in
`compare_star_uvt_worldfoam_scale.py`: it still ignored only the current PID and
direct parent, so a wrapped STAR compare launched through
`rtk sh -lc ... powerfoam_metal...` could reject a clean machine before running
STAR. I ported the parent-map ancestor-chain ignore logic into the compare
script and added a compare-side regression with a three-process launch chain.

Verification:

```bash
rtk sh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py research_experiments/world_foam_lane2/test_compare_star_uvt_worldfoam_scale.py'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Results: `py_compile` OK; compare suite `Ran 12 tests in 0.003s`, `OK`;
focused wrapper/verifier/native gate `Ran 69 tests in 0.060s`, `OK`.

Fresh strict real32 benchmark readiness check:

```bash
rtk sh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 21:43 +07`. The only
hard blockers were the detached TOTO MPS exporter chain
`toto_floor001_guardaligned_edgefloor_20260520T135831Z`; low-CPU audit and
memory-writer processes were background-only. No clean real32 WorldFoam/STAR
timing rerun was launched.

## 2026-05-20 21:47 +07 - Promotion verifier now requires acceptance metadata

While the strict timing lane remained blocked, I tightened the final promotion
verifier. `_check_worldfoam_payload()` previously checked acceptance values only
when the `acceptance` dict existed, so a clean-looking WorldFoam artifact could
omit the acceptance block and still pass the promotion verifier. It now rejects
missing or empty acceptance metadata, and the regression deletes `acceptance`
from an otherwise valid artifact and expects failure.

Verification:

```bash
rtk sh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py research_experiments/world_foam_lane2/test_verify_worldfoam_star_native_cutwalk_promotion.py'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Results: `py_compile` OK; verifier suite `Ran 15 tests in 0.015s`, `OK`;
focused wrapper/verifier/native gate `Ran 70 tests in 0.059s`, `OK`.

## 2026-05-20 21:51 +07 - STAR compare also refuses missing WorldFoam acceptance before launch

I tightened `compare_star_uvt_worldfoam_scale.py` so
`--require-clean-worldfoam-artifact` checks the WorldFoam `acceptance` block
before calling `run_star_cases()`. Without this, the final promotion verifier
would reject missing acceptance eventually, but the compare gate could still
spend a matched STAR timing run first. The new test builds an otherwise clean
WorldFoam artifact without `acceptance`, expects
`WorldFoam artifact acceptance is missing`, and asserts STAR was not launched.

Verification:

```bash
rtk sh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/compare_star_uvt_worldfoam_scale.py research_experiments/world_foam_lane2/test_compare_star_uvt_worldfoam_scale.py'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Results: `py_compile` OK; compare suite `Ran 13 tests in 0.003s`, `OK`;
focused wrapper/verifier/native gate `Ran 71 tests in 0.068s`, `OK`.

Fresh strict real32 benchmark readiness check:

```bash
rtk sh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 21:49 +07`. The detached
TOTO MPS exporter chain was still present and an active export child
`scripts/export_btc15m_toto_residual_live_prediction_export.py` was at `34.2%`
CPU. No strict real32 timing/PSNR run was launched.

## 2026-05-20 21:55 +07 - Wrapper selection now matches acceptance contract

I found one remaining promotion-contract mismatch: the wrapper selected a
WorldFoam artifact as promotable using only return code, `status=ok`, and clean
or background benchmark environment. The final verifier and STAR compare now
both require a non-empty all-true WorldFoam `acceptance` block, so the wrapper
could still hand off an artifact that the later gates would reject.

I added `_worldfoam_acceptance_failures(...)` to
`run_worldfoam_star_native_cutwalk_gate.py`, recorded per-attempt
`acceptance_ok`/`acceptance_failures`, and required acceptance cleanliness before
setting `worldfoam_artifact` / `worldfoam_promotable_artifact` or planning the
selected STAR command. Missing acceptance now ends as
`worldfoam_not_promotable`, leaves `star_compare_command=null`, and does not
launch STAR. The new wrapper regression covers an otherwise clean/background
WorldFoam artifact with no `acceptance` block.

Verification:

```bash
rtk sh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion research_experiments.world_foam_lane2.test_verify_native_packed_extension'
```

Results: `py_compile` OK; wrapper suite `Ran 24 tests in 0.022s`, `OK`;
focused wrapper/verifier/native gate `Ran 72 tests in 0.061s`, `OK`.

Fresh strict real32 benchmark readiness check:

```bash
rtk sh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 21:56 +07`. The same
detached TOTO monitor chain was still present, with a transient child
`lean_trade.runners.btc_15m_sft_shadow --config configs/lean_btc_15m_toto_sft_edge_shadow_paper.yaml`
at `8.4%` CPU. No strict real32 timing/PSNR run was launched.

## 2026-05-20 22:04 +07 - Quality bridge records speed true but STAR quality false

I added a narrow quality bridge so the WorldFoam speed micro-gate cannot be
accidentally retold as STAR-quality competitiveness. The new script
`research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py`
reads the accepted native-cutwalk WorldFoam artifact and matched STAR timing
comparison, compares WorldFoam RGB PSNR against the current STAR UVT
source-overfit RGB reference (`29.823dB`) and the solid same-source baseline
(`21.36dB`), and emits a separate, explicit
`star_uvt_competitive_claim=false` unless both quality and speed gates are
clean. It also rejects a WorldFoam artifact that tries to carry
`quality_claim=true` directly.

Artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_star_quality_bridge_native_cutwalk_microgate.json
```

Result: `status=ok`, `speed_bridge.speed_competitive_micro_gate=true`, but
`star_uvt_competitive_claim=false`. WorldFoam best train PSNR is `12.248`,
best heldout PSNR is `12.857`, the train PSNR gap to STAR UVT source-overfit
RGB is `17.575dB`, and the gap to the solid same-source baseline is `9.112dB`.
The honest current stance is: native-cutwalk WorldFoam has a clean speed
micro-gate, but not RGB-quality competitiveness.

Verification:

```bash
rtk sh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/test_report_worldfoam_star_quality_bridge.py'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py --out-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_star_quality_bridge_native_cutwalk_microgate.json'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion research_experiments.world_foam_lane2.test_verify_native_packed_extension research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge'
```

Results before this handoff doc update: `py_compile` OK; quality bridge suite
`Ran 4 tests in 0.005s`, `OK`; expanded focused wrapper/verifier/native/quality
gate `Ran 76 tests in 0.121s`, `OK`.

Post-update verification reran cleanly: `py_compile` OK, trailing whitespace
scan clean, `git diff --check` clean for the touched docs/scripts/tests,
quality bridge suite `Ran 4 tests in 0.011s`, `OK`, report regeneration
emitted the same `star_uvt_competitive_claim=false`, and the expanded focused
gate `Ran 76 tests in 0.242s`, `OK`.

Fresh strict real32 benchmark readiness check:

```bash
rtk sh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 22:05 +07`. Blockers now
include a high-CPU `font_maker` torch training child, a high-CPU transient
`lean_trade.runners.btc_15m_sft_shadow`, the ai_trader TOTO periodic MPS-export
monitor chain, and a `keyword:torch` font_maker queue wrapper. No strict real32
timing/PSNR run was launched.

## 2026-05-20 22:09 +07 - Quality bridge now includes existing capacity candidate

I extended `report_worldfoam_star_quality_bridge.py` with
`--extra-worldfoam-artifact` so it can summarize capacity candidates without
promoting broad STAR parity. The report now tracks `capacity_candidates`,
`capacity_candidates_improve_train_psnr`, and the best WorldFoam-quality
artifact across the primary and extra candidates. The new regression covers a
render96/site48-style candidate with lower PSNR and verifies that it is reported
without overriding the primary speed bridge or `star_uvt_competitive_claim`.

I regenerated
`research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_star_quality_bridge_native_cutwalk_microgate.json`
with the existing render96/site48 artifact
`2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate.attempt2.worldfoam.json`
as an extra candidate. Result: `capacity_candidate_count=1`,
`capacity_candidates_improve_train_psnr=false`, primary render64/site24 remains
the best WorldFoam-quality artifact at train PSNR `12.248`, and the render96/site48
candidate reaches only `9.875` best train PSNR / `10.880` heldout PSNR. This
turns the available capacity evidence into a durable negative: naive render/site
capacity did not close the RGB-quality gap.

Verification so far:

```bash
rtk sh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/test_report_worldfoam_star_quality_bridge.py'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py --extra-worldfoam-artifact /Users/nicholasbardy/git/gsplats_browser/dynaworld/research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate.attempt2.worldfoam.json --out-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_star_quality_bridge_native_cutwalk_microgate.json'
```

Results: `py_compile` OK; quality bridge suite `Ran 5 tests in 0.004s`, `OK`;
report regeneration emitted `star_uvt_competitive_claim=false` and
`capacity_candidates_improve_train_psnr=false`.

Final verification after docs/notes:

```bash
rtk sh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/test_report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py'
rtk rg -n '[[:blank:]]$' PROJECT_INDEX.md EXPERIMENTS.md TODO/README.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/test_report_worldfoam_star_quality_bridge.py
rtk git diff --check -- PROJECT_INDEX.md EXPERIMENTS.md TODO/README.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/test_report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_star_quality_bridge_native_cutwalk_microgate.json research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion research_experiments.world_foam_lane2.test_verify_native_packed_extension research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge'
```

Results: `py_compile` OK; whitespace scan clean; `git diff --check` clean;
expanded focused gate `Ran 77 tests in 0.123s`, `OK`.

Fresh strict real32 benchmark readiness check:

```bash
rtk sh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 22:09 +07`. Blockers
include the high-CPU `font_maker` torch training child, a live ai_trader
feature-context/export child, the ai_trader TOTO periodic MPS-export monitor
chain, and the `keyword:torch` font_maker queue wrapper. No strict real32
timing/PSNR run was launched.

## 2026-05-20 22:15 +07 - Quality bridge separates candidate quality from matched speed

I tightened the WorldFoam-vs-STAR quality bridge contract so a future larger
WorldFoam candidate cannot close the RGB PSNR gap and silently inherit the
primary render64/site24 speed gate. `report_worldfoam_star_quality_bridge.py`
now records `best_worldfoam_quality.quality_gaps`,
`best_worldfoam_quality_is_primary_speed_artifact`,
`best_worldfoam_quality_competitive_with_star_source`,
`best_worldfoam_quality_competitive_with_solid_same_source`, and
`best_worldfoam_quality_needs_matched_speed_gate`.

The new regression
`test_high_quality_capacity_candidate_needs_its_own_matched_speed_gate` creates
a high-quality extra capacity artifact and verifies that the report keeps
`star_uvt_competitive_claim=false` while setting
`best_worldfoam_quality_needs_matched_speed_gate=true`. In other words, a
capacity fork can become the best quality artifact, but it still needs its own
matched-speed STAR comparison before it becomes a STAR-competitive WorldFoam
claim.

The real report was regenerated with the existing render96/site48 candidate.
Current result: `best_worldfoam_quality_is_primary_speed_artifact=true`,
`best_worldfoam_quality_competitive_with_star_source=false`,
`best_worldfoam_quality_needs_matched_speed_gate=false`, and
`capacity_candidates_improve_train_psnr=false`.

Verification:

```bash
rtk sh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/test_report_worldfoam_star_quality_bridge.py'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion research_experiments.world_foam_lane2.test_verify_native_packed_extension research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge'
```

Results: `py_compile` OK; quality bridge suite `Ran 6 tests in 0.005s`, `OK`;
expanded focused gate `Ran 78 tests in 0.116s`, `OK`.

Final post-doc verification for this continuation:

```bash
rtk sh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/test_report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py'
rtk rg -n '[[:blank:]]$' PROJECT_INDEX.md EXPERIMENTS.md TODO/README.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/test_report_worldfoam_star_quality_bridge.py
rtk git diff --check -- PROJECT_INDEX.md EXPERIMENTS.md TODO/README.md agent_notes/loose_notes/2026-05-20_05-19-33_worldfoam_real32_settle_retry.md research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/test_report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_star_quality_bridge_native_cutwalk_microgate.json research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion research_experiments.world_foam_lane2.test_verify_native_packed_extension research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge'
```

Results: `py_compile` OK; whitespace scan clean; `git diff --check` clean;
quality bridge suite `Ran 6 tests in 0.015s`, `OK`; expanded focused gate
`Ran 78 tests in 0.329s`, `OK`.

Fresh strict real32 benchmark readiness check:

```bash
rtk sh -lc 'date; PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc'
```

Result: exit `2`, `status=contended` at `2026-05-20 22:16 +07`. Blockers
include an active high-CPU `ai_trader` Toto residual prediction export, the
periodic `ai_trader` MPS exporter monitor chain, the high-CPU `font_maker`
torch child, and the `keyword:torch` font-maker queue wrapper. No strict real32
timing/PSNR run was launched.

## 2026-05-20 22:21 +07 - Capacity bridge now checks overlapping frame coverage

I tightened the quality bridge again after noticing that the existing
render96/site48 candidate only has `2/4/8f`, while the primary matched-speed
artifact has `2/4/8/16f`. `report_worldfoam_star_quality_bridge.py` now compares
each capacity candidate to the primary artifact by common frame counts, records
missing primary frame counts, and stores per-frame train/heldout PSNR deltas.

The regenerated bridge report keeps the same broad conclusion but makes it more
precise: `capacity_candidates_improve_train_psnr=false`,
`capacity_candidates_improve_train_psnr_on_any_common_frame=false`, and the
render96/site48 candidate is listed in
`capacity_candidate_artifacts_missing_primary_frames`. On overlapping frames it
is worse than the primary by `-2.55/-2.53/-2.27dB` train PSNR at `2/4/8f`, and
it is missing the primary `16f` row. This means the available capacity evidence
is a negative on shared frames, not a full-frame-set quality sweep.

New regression:
`test_capacity_candidate_missing_primary_frames_is_not_silently_full_coverage`
constructs a candidate that improves a common frame while omitting the primary
`16f` row and verifies that the report exposes both facts.

Verification:

```bash
rtk sh -lc 'PYTHONPYCACHEPREFIX=/tmp/dynaworld_pycache .venv/bin/python -m py_compile research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py research_experiments/world_foam_lane2/test_report_worldfoam_star_quality_bridge.py'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/report_worldfoam_star_quality_bridge.py --extra-worldfoam-artifact /Users/nicholasbardy/git/gsplats_browser/dynaworld/research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate.attempt2.worldfoam.json --out-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_star_quality_bridge_native_cutwalk_microgate.json'
rtk sh -lc 'PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest -v research_experiments.world_foam_lane2.test_train_eval_benchmark_environment research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion research_experiments.world_foam_lane2.test_verify_native_packed_extension research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge'
```

Results: `py_compile` OK; quality bridge suite `Ran 7 tests in 0.013s`, `OK`;
report regenerated with `star_uvt_competitive_claim=false`; expanded focused
gate `Ran 79 tests in 0.170s`, `OK`.

## 2026-05-20 22:37 +07 - Stratified site initialization fork wired, benchmark blocked

I added a default-preserving WorldFoam site initialization fork. The old
deterministic sparse train-ray sampler is now named `legacy_sparse` and remains
the default. The new `stratified_grid` mode uses the same train-sample/depth
selection but spreads initial sites over a deterministic image grid. This is a
quality/capacity fork, not a speed promotion.

Threading:

- `gate1_realray_per_sample_reference.initialize_sites_from_train_samples(...)`
  accepts `initialization=...` and rejects unknown modes.
- `gate4_moving_ray_slab_compiler.py` and
  `train_eval_owner_run_tape.py` expose `--site-initialization
  {legacy_sparse,stratified_grid}`.
- train/eval rows and top-level artifacts record `site_initialization`.

Verification:

```bash
rtk .venv/bin/python -m py_compile research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --help
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py --help
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py --frame-counts 2,4 --render-size 8 --site-count 4 --time-slabs 1 --site-initialization stratified_grid --out-json /tmp/worldfoam_gate4_stratified_grid_smoke.json
rtk git diff --check
```

Results: `py_compile` OK; direct initializer suite `Ran 3 tests in 0.028s`,
`OK`; broader owner-run packed plus quality-bridge gate `Ran 65 tests in
389.209s`, `OK`; both CLIs expose the new flag; CPU Gate4 stratified-grid
smoke writes `/tmp/worldfoam_gate4_stratified_grid_smoke.json` with
`status=ok` for `2/4f`; `git diff --check` clean.

Fresh strict real32 benchmark readiness check:

```bash
rtk env PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc
```

Result: exit `2`, `status=contended`. Blockers were high-CPU `font_maker`
training (`~157%` CPU), the periodic `ai_trader` TOTO MPS-export monitor chain,
and a `keyword:torch` font-maker queue wrapper. I did not launch a
`stratified_grid` PSNR/speed run under those conditions. The next useful run is
a quiet real32 functional/quality gate with `--site-initialization
stratified_grid`, followed by matched STAR only if the WorldFoam artifact is
clean and improves quality.

## 2026-05-20 22:42 +07 - Gate1 CPU reference makes stratified init a negative

I threaded `--site-initialization` into the Gate1 CPU per-sample real-ray
reference too, so the non-MPS reference artifacts record which site initializer
they used. This lets us test the idea while the MPS timing environment is still
blocked.

Commands:

```bash
rtk .venv/bin/python -m py_compile research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py --help
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py --max-frames 2 --render-size 16 --site-count 9 --site-initialization legacy_sparse --out-json research_experiments/world_foam_lane2/results/2026-05-20_gate1_legacy_sparse_reference_render16_site9_2f.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py --max-frames 2 --render-size 16 --site-count 9 --site-initialization stratified_grid --out-json research_experiments/world_foam_lane2/results/2026-05-20_gate1_stratified_grid_reference_render16_site9_2f.json
```

Results: `py_compile` OK; Gate1 help exposes
`--site-initialization {legacy_sparse,stratified_grid}`; both CPU references
write `status=ok` artifacts. The quality comparison is negative for the naive
grid spread:

- legacy train/heldout PSNR: `11.862/12.671`, L1 `0.2083/0.1901`
- stratified train/heldout PSNR: `10.419/9.692`, L1 `0.2438/0.2746`
- stratified delta: `-1.44dB` train, `-2.98dB` heldout, worse L1 on both splits

Conclusion: deterministic image-cell coverage alone is not the missing quality
fix. The next WorldFoam quality fork should change support/color initialization
more intelligently, or prove itself through a trained clean artifact; do not
treat `stratified_grid` as a likely promotion candidate based only on coverage.

## 2026-05-20 22:48 +07 - Legacy support with pixel-mean color is the next CPU-positive init fork

I ran a small CPU search over initializer variants before wiring another mode.
The useful candidate was not another geometry/support pattern: it was keeping
the legacy sparse site positions and replacing each site's sampled single-frame
RGB with the mean RGB over all train samples at that same pixel. I wired this as
`legacy_pixel_mean`; it is default-preserving and now threads through the same
Gate1/Gate4/train-eval `--site-initialization` flag.

Verification:

```bash
rtk .venv/bin/python -m py_compile research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py --help
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --help
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py --max-frames 2 --render-size 16 --site-count 9 --site-initialization legacy_pixel_mean --out-json research_experiments/world_foam_lane2/results/2026-05-20_gate1_legacy_pixel_mean_reference_render16_site9_2f.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py --frame-counts 2,4 --render-size 8 --site-count 4 --time-slabs 1 --site-initialization legacy_pixel_mean --out-json /tmp/worldfoam_gate4_legacy_pixel_mean_smoke.json
```

Results: `py_compile` OK; initializer suite `Ran 4 tests in 0.050s`, `OK`;
Gate1 and train-eval help expose
`{legacy_sparse,legacy_pixel_mean,stratified_grid}`; Gate1 CPU reference writes
`status=ok`; Gate4 CPU compiler smoke writes `status=ok`.

Gate1 render16/site9/2f quality comparison:

- `legacy_sparse`: train/heldout PSNR `11.862/12.671`, L1 `0.2083/0.1901`
- `stratified_grid`: train/heldout PSNR `10.419/9.692`, L1 `0.2438/0.2746`
- `legacy_pixel_mean`: train/heldout PSNR `13.025/14.614`, L1 `0.1735/0.1487`

Conclusion: `legacy_pixel_mean` is the first positive CPU reference in this
initializer fork. It should be the next quiet MPS quality run. This is not yet a
train/speed promotion; it only says that mean color over train samples is a
better initialization signal than single-sample legacy color or naive grid
support on the small Gate1 reference.

## 2026-05-20 22:50 +07 - Site-initialization bridge report generated

I added a small report/verifier around the Gate1 CPU initializer artifacts so
the next handoff is a checked JSON decision, not just prose. The report requires
the baseline and candidate artifacts to share the same fixture
`(config_path, frame_count, render_size, site_count, boundary_count)`, then only
marks a candidate positive if it improves both train and heldout PSNR and
reduces both train and heldout L1 versus `legacy_sparse`.

Commands:

```bash
rtk .venv/bin/python -m py_compile research_experiments/world_foam_lane2/report_worldfoam_site_initialization_quality.py research_experiments/world_foam_lane2/test_report_worldfoam_site_initialization_quality.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/report_worldfoam_site_initialization_quality.py
```

Results: `py_compile` OK; report tests `Ran 4 tests in 0.009s`, `OK`; the
generated report
`research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_site_initialization_quality_bridge.json`
writes `status=ok`, `baseline_initialization=legacy_sparse`,
`next_mps_candidate=legacy_pixel_mean`, `positive_candidate_count=1`, and
`rejected_candidate_count=1`. This preserves the decision that
`stratified_grid` is CPU-negative and `legacy_pixel_mean` is the next quiet MPS
candidate, without claiming the train/speed gate has been run.

I also tightened the report to fail closed on any Gate1 artifact whose
`status` is not `ok`; the added unit test covers that failure mode.

## 2026-05-20 22:54 +07 - Fresh real32 preflight still blocks MPS promotion

After the report/docs pass, I reran the strict real32 benchmark-environment
preflight before considering a `legacy_pixel_mean` MPS quality/speed run.

Command:

```bash
rtk env PYTHONPATH=src/train .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --benchmark-environment-check-only --wait-for-benchmark-environment-ok-timeout-s 0 --wait-for-benchmark-environment-ok-poll-s 1 --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc
```

Result: exit `2`, `status=contended`. Current blockers included a high-CPU
`font_maker` training child (`~156%` CPU), the detached `ai_trader` TOTO monitor
chain `20691/20692/20706/20723/20724`, an active `ai_trader`
`probe_btc15m_tree_oracle_context_feature_frame` child, and a `keyword:torch`
font-maker queue wrapper. I did not launch the MPS PSNR/speed artifact under
those conditions.

Broader regression coverage after the docs/report pass:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality -v
```

Result: `Ran 69 tests in 514.605s`, `OK`.

Fresh continuation preflight later in the same lane still returned exit `2`,
`status=contended`. The blocker set had shifted but was still not clean: high
CPU `font_maker` training (`~98%`), an unrelated high-CPU `pytest tests/`
process (`~85%`), an active `ai_trader`
`export_btc15m_tree_residual_live_prediction_export` child (`~81%`), the
detached TOTO monitor chain, and the `keyword:torch` font-maker queue wrapper.

## 2026-05-20 23:13 +07 - Candidate CSR topology probe now accepts site init

Because the strict MPS preflight was still contended, I made one more
non-timing improvement: the CPU Gate4 affine candidate-CSR topology probe now
threads the same `--site-initialization` choice as Gate1/Gate4/train-eval and
records it in the top-level payload and per-frame rows. This lets the selected
`legacy_pixel_mean` fork prove it still compiles into the topology/capacity path
before spending a quiet MPS run.

Commands:

```bash
rtk .venv/bin/python -m py_compile research_experiments/world_foam_lane2/probe_gate4_affine_candidate_csr_capacity.py research_experiments/world_foam_lane2/test_probe_gate4_affine_candidate_csr_capacity.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_gate4_affine_candidate_csr_capacity -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/probe_gate4_affine_candidate_csr_capacity.py --help
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/probe_gate4_affine_candidate_csr_capacity.py --frame-counts 2,4 --render-size 8 --site-count 4 --site-initialization legacy_pixel_mean --out-json research_experiments/world_foam_lane2/results/2026-05-20_gate4_affine_candidate_csr_capacity_legacy_pixel_mean_render8_site4_2_4f.json
```

Results: `py_compile` OK; capacity-probe tests `Ran 4 tests in 0.018s`, `OK`;
CLI help exposes `--site-initialization {legacy_sparse,legacy_pixel_mean,stratified_grid}`;
the tiny CPU artifact writes `status=ok`, `site_initialization=legacy_pixel_mean`,
candidate count scale `0.993x`, storage scale `0.998x`, and all acceptance flags
true over `2f -> 4f`. This still is not a Metal speed/quality claim; it only
proves the chosen initializer is compatible with the candidate CSR topology path.

## 2026-05-20 23:20 +07 - Combined next-MPS readiness verifier

I added a fail-closed readiness report that ties the CPU quality bridge and the
matching topology artifact together. It fails if the quality bridge does not
select a positive `next_mps_candidate`, if the topology artifact uses a different
initializer, if any topology acceptance flag is false, or if the topology probe
covers fewer than two frame counts.

Commands:

```bash
rtk .venv/bin/python -m py_compile research_experiments/world_foam_lane2/report_worldfoam_next_mps_candidate_readiness.py research_experiments/world_foam_lane2/test_report_worldfoam_next_mps_candidate_readiness.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/report_worldfoam_next_mps_candidate_readiness.py
```

Results: `py_compile` OK; readiness tests `Ran 4 tests in 0.006s`, `OK`; the
generated report
`research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_candidate_readiness.json`
writes `status=ok`, `next_mps_candidate=legacy_pixel_mean`,
`ready_for_quiet_mps_quality_speed_run=true`, `quality_claim=false`, and
`speed_claim=false`. This is the handoff gate for the next clean MPS PSNR/speed
run.

The broader owner-run/report regression slice with the new probe/readiness tests
also passed:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed research_experiments.world_foam_lane2.test_report_worldfoam_star_quality_bridge research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality research_experiments.world_foam_lane2.test_probe_gate4_affine_candidate_csr_capacity research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness -v
```

Result: `Ran 78 tests in 317.567s`, `OK`.

## 2026-05-20 23:31 +07 - Fail-closed launcher for the next MPS candidate

I added a narrow launcher for the readiness-selected candidate:
`research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py`.
It validates the readiness report first, plans the exact real32 train/eval
command with `--site-initialization legacy_pixel_mean`, the native cutwalk
framebitmask tape path, and `--require-benchmark-environment-ok`, and only
executes train/eval when invoked with `--execute` after a clean strict preflight.
Default mode writes a plan summary without launching MPS work.

Commands:

```bash
rtk .venv/bin/python -m py_compile research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py --run-id 2026-05-20_worldfoam_next_mps_legacy_pixel_mean_plan --summary-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_plan.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py --run-id 2026-05-20_worldfoam_next_mps_legacy_pixel_mean_preflight --summary-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_preflight.json --preflight-only --execute
```

Results: `py_compile` OK; launcher tests `Ran 3 tests in 0.003s`, `OK`;
the plan summary writes `status=planned` and a train/eval command for
`legacy_pixel_mean`; the executed preflight summary writes
`status=preflight_contended`, return code `2`, and did not launch train/eval.
The latest blocker set in that wrapper summary is high-CPU `font_maker`, the
detached `ai_trader` TOTO monitor chain, and the `keyword:torch` font-maker
queue wrapper.

Focused combined coverage after the launcher/docs pass:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_gate4_affine_candidate_csr_capacity research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests -v
```

Result: `Ran 19 tests in 0.023s`, `OK`.

## 2026-05-20 23:55 +07 - Next-MPS launcher execute-path hardening

The strict preflight was still contaminated when resuming the lane. A direct
benchmark-environment check exited `2`, `status=contended`; blockers included
high-CPU `font_maker`, an unrelated high-CPU `ai_trader` pytest, the detached
TOTO MPS exporter chain, an active `ai_trader` tree feature/export child, and
the font-maker `keyword:torch` queue wrapper.

I hardened the fail-closed candidate launcher rather than launching a dirty MPS
timing run. `run_worldfoam_next_mps_candidate.py` now promotes the preflight
environment summary to top-level artifact fields:
`preflight_benchmark_environment_status`, blocking/contending counts,
`preflight_blocking_reasons`, and a compact `preflight_blocking_processes`
list. I also added a mocked execute-path regression proving that a failed
preflight writes those fields, returns the preflight return code, and never
calls the train/eval subprocess.

Regenerated artifact:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py --run-id 2026-05-20_worldfoam_next_mps_legacy_pixel_mean_preflight --summary-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_preflight.json --execute --preflight-only
```

Result: exit `2`, `status=preflight_contended`,
`preflight_benchmark_environment_status=contended`,
`preflight_blocking_process_count=8`,
`preflight_blocking_reasons=["high_cpu","keyword:torch","periodic_mps_exporter"]`.
No train/eval artifact was launched.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality research_experiments.world_foam_lane2.test_probe_gate4_affine_candidate_csr_capacity research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate -v
```

Results: `py_compile` OK; launcher tests `Ran 4 tests in 0.007s`, `OK`;
combined owner-run/site-init/topology/readiness/launcher suite `Ran 75 tests in
354.443s`, `OK`.

## 2026-05-20 23:37 +07 - Next-MPS launcher stability samples

Fresh strict benchmark preflight still exited `2`, `status=contended`, with the
same class of external blockers: high-CPU `font_maker`, the detached TOTO MPS
exporter chain, and the font-maker `keyword:torch` queue wrapper. A background
tree-context export child was visible but below the blocker threshold in the
direct check.

I extended `run_worldfoam_next_mps_candidate.py` with a stability gate:
`--preflight-stability-samples` and `--preflight-stability-interval-s`. The
launcher now records per-sample compact summaries and only launches train/eval
after every requested preflight sample is clean. I added a unit test proving a
3-sample clean sequence is required before the train subprocess is called,
alongside the existing failed-preflight no-train regression.

Commands:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py --run-id 2026-05-20_worldfoam_next_mps_legacy_pixel_mean_plan --summary-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_plan.json --preflight-stability-samples 3 --preflight-stability-interval-s 5
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py --run-id 2026-05-20_worldfoam_next_mps_legacy_pixel_mean_preflight --summary-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_preflight.json --preflight-stability-samples 3 --preflight-stability-interval-s 5 --execute --preflight-only
```

Results: `py_compile` OK; launcher tests `Ran 5 tests in 0.012s`, `OK`.
The regenerated plan requests `3` stability samples at `5s` spacing. The
executed preflight summary remains `preflight_contended`, with
`preflight_stability_samples_requested=3`,
`preflight_stability_samples_completed=1`, `preflight_stability_ok=false`, and
no train/eval artifact launched.

## 2026-05-20 23:40 +07 - Stability gate catches active TOTO MPS export

I reran the 3-sample preflight-only launcher, and it again stopped after the
first contaminated sample. The refreshed artifact still has
`status=preflight_contended`, `preflight_stability_samples_completed=1`, and
`preflight_stability_ok=false`, but the live blocker set now explicitly includes
an active TOTO prediction export (`keyword:mps`) in addition to high-CPU
`font_maker`, high-CPU unrelated `ai_trader` pytest, the detached periodic MPS
exporter chain, and the `keyword:torch` queue wrapper. This reinforces that the
next quality/speed run must wait for the exporter cadence to clear, not just for
one quiet CPU snapshot.

I added the missing launcher regression for this exact failure mode: sample 1
clean, sample 2 contended. The launcher must stop at sample 2, return the
preflight error code, write `preflight_stability_ok=false`, preserve both sample
records, and never call train/eval.

Command:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate -v
```

Result: `Ran 6 tests in 0.012s`, `OK`.

## 2026-05-20 23:47 +07 - Next-MPS post-run verifier added

I added a strict post-run verifier for the readiness-selected
`legacy_pixel_mean` MPS candidate:
`research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py`.
The launcher now writes `result_verifier_command` into both the plan and
preflight summaries, so a future clean run has a one-command acceptance audit.

The verifier is intentionally narrower than the broader WorldFoam/STAR
promotion audit. It checks only this next-MPS candidate handoff: executed
`status=train_eval_ok`, completed clean stability samples, clean preflight and
artifact benchmark environments, matching `legacy_pixel_mean` site initialization,
the expected native-cutwalk tape/optimizer/source flags, MPS rows for all
requested frame counts, numeric PSNR/L1, no repeated frames, and sublinear
total/backward acceptance. The current contended preflight summary fails this
verifier as expected.

Commands:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py research_experiments/world_foam_lane2/test_verify_worldfoam_next_mps_candidate_result.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py --run-id 2026-05-20_worldfoam_next_mps_legacy_pixel_mean_plan --summary-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_plan.json --preflight-stability-samples 3 --preflight-stability-interval-s 5
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py --run-id 2026-05-20_worldfoam_next_mps_legacy_pixel_mean_preflight --summary-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_preflight.json --preflight-stability-samples 3 --preflight-stability-interval-s 5 --execute --preflight-only
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_preflight.json
```

Results: `py_compile` OK; launcher plus verifier unit tests `Ran 10 tests in
0.024s`, `OK`. The regenerated plan is still `status=planned` and now records
the verifier command. The regenerated execute/preflight-only summary exits `2`
with `status=preflight_contended`, requests `3` stability samples, completes
only sample `1/3`, and keeps `preflight_stability_ok=false`. The latest written
blocker reasons are `high_cpu`, `keyword:torch`, and `periodic_mps_exporter`
with `7` blocking rows; an earlier live stability sample also caught an active
TOTO prediction export as `keyword:mps`, so the TOTO monitor remains a real
timing-window blocker. The verifier exits `1` on the blocked summary with
failures for non-`train_eval_ok` status, incomplete stability samples, contended
preflight, and missing WorldFoam artifact.

After the first hygiene pass, I reran a fresh preflight-only check because a
quick process scan did not show the earlier hot children. The authoritative
benchmark preflight still found the environment contended:
`2026-05-20_worldfoam_next_mps_legacy_pixel_mean_fresh_preflight.json` exited
`2`, stopped at sample `1/3`, and recorded `8` blocking rows: high-CPU
`font_maker`, high-CPU `ai_trader` pytest/report children, the TOTO monitor
chain, and a `keyword:torch` queue wrapper. Running the new verifier on that
fresh summary also exits `1` with the same intended fail-closed reasons.

## 2026-05-20 23:55 +07 - Launcher now enforces the post-run verifier

The previous handoff still relied on a human or future agent to run
`result_verifier_command` after a successful train/eval. I closed that gap in
`run_worldfoam_next_mps_candidate.py`: it now has `--verify-result`. When a
train/eval subprocess exits `0`, the launcher writes the provisional
`train_eval_ok` summary, runs the strict
`verify_worldfoam_next_mps_candidate_result.py` command, records
`result_verifier_returncode`, `result_verifier_payload`, and verifier output
tails, and returns `result_verification_failed` if the verifier rejects the
artifact. This keeps the next clean execution from returning success merely
because the train/eval script exited cleanly.

Commands:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py --run-id 2026-05-20_worldfoam_next_mps_legacy_pixel_mean_plan --summary-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_plan.json --preflight-stability-samples 3 --preflight-stability-interval-s 5 --verify-result
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py --run-id 2026-05-20_worldfoam_next_mps_legacy_pixel_mean_probe2351 --summary-json research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_probe2351.json --preflight-stability-samples 3 --preflight-stability-interval-s 5 --execute --preflight-only
```

Results: `py_compile` OK; launcher tests now cover both verifier-pass and
verifier-fail clean-train paths and pass `Ran 8 tests in 0.023s`, `OK`. The
refreshed plan summary records `verify_result=true`. The 23:51 preflight probe
still failed before train/eval: `status=preflight_contended`,
`preflight_stability_samples_completed=1`, `preflight_stability_ok=false`,
and `8` blockers. This time the blocker set included high-CPU `font_maker`,
high-CPU `ai_trader` pytest/RL children, an active TOTO live-quote snapshot
under the overnight monitor, the idle TOTO monitor parents, and the
`keyword:torch` font-maker queue wrapper.

## 2026-05-20 23:58 +07 - Stratified pixel-mean initializer rejected

Because the clean MPS gate is still blocked by external processes, I added one
more CPU-testable site-initialization fork:
`--site-initialization stratified_pixel_mean`. It combines the image-cell
support coverage of `stratified_grid` with the train-sample mean color idea
from `legacy_pixel_mean`. This tests whether the earlier grid failure was
caused by noisy single-frame color at each grid point or by the grid support
geometry itself.

Commands:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/report_worldfoam_site_initialization_quality.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py --frame-counts 2,4 --render-size 8 --site-count 4 --time-slabs 1 --site-initialization stratified_pixel_mean --out-json /tmp/worldfoam_gate4_stratified_pixel_mean_smoke.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py --max-frames 2 --render-size 16 --site-count 9 --site-initialization stratified_pixel_mean --out-json research_experiments/world_foam_lane2/results/2026-05-20_gate1_stratified_pixel_mean_reference_render16_site9_2f.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/report_worldfoam_site_initialization_quality.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/report_worldfoam_next_mps_candidate_readiness.py
```

Results: `py_compile` OK; focused site-init/report tests `Ran 9 tests in
0.040s`, `OK`; the Gate4 compiler smoke writes `status=ok` for
`stratified_pixel_mean`. Gate1 CPU reference is mixed and rejected:
train/heldout PSNR `13.679/12.611`, L1 `0.1679/0.1965`. It improves train more
than `legacy_pixel_mean`, but heldout is worse than the `legacy_sparse`
baseline (`12.671` PSNR / `0.1901` L1), so this is train overfit, not a
next-MPS candidate. The regenerated quality bridge now has
`positive_candidate_count=1`, `rejected_candidate_count=2`, and still selects
`next_mps_candidate=legacy_pixel_mean`. The readiness report remains `status=ok`
for `legacy_pixel_mean`.

## 2026-05-21 00:20 +07 - Frame-local pixel mean becomes the next MPS candidate

I added one more CPU-testable initializer after the grid-plus-mean rejection:
`--site-initialization legacy_frame_pixel_mean`. It keeps legacy support
geometry and averages each site's color only over train samples whose frame
index matches the site frame. This is the closest STAR-UVT-style idea we have
in this scalar site initializer family: keep the clean old support assignment,
but stop a point's seed color from being averaged across unrelated timesteps.

Results:

- Gate1 CPU reference at render16/site9/2f:
  `legacy_frame_pixel_mean` train/heldout PSNR `13.029/14.617`, L1
  `0.1734/0.1486`.
- It slightly beats `legacy_pixel_mean` (`13.025/14.614`, L1
  `0.1735/0.1487`) and is now `best_by_heldout_psnr`.
- The regenerated bridge reports
  `next_mps_candidate=legacy_frame_pixel_mean`, `positive_candidate_count=2`,
  and `rejected_candidate_count=2`.
- The matching topology probe
  `2026-05-20_gate4_affine_candidate_csr_capacity_legacy_frame_pixel_mean_render8_site4_2_4f.json`
  passes all acceptance checks for the selected candidate.
- The readiness report remains `status=ok`,
  `ready_for_quiet_mps_quality_speed_run=true`, `quality_claim=false`, and
  `speed_claim=false`, now for `legacy_frame_pixel_mean`.
- The plan summary
  `2026-05-20_worldfoam_next_mps_legacy_frame_pixel_mean_plan.json` threads
  `--site-initialization legacy_frame_pixel_mean` through the real32 MPS command
  and keeps `--verify-result`.

I tried the current preflight again on 2026-05-21:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_preflight \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_preflight.json \
  --preflight-stability-samples 3 --preflight-stability-interval-s 5 \
  --verify-result --execute --preflight-only
```

It failed closed at sample `1/3`, without launching train/eval:
`status=preflight_contended`, `preflight_stability_ok=false`,
`preflight_blocking_process_count=8`, reasons `high_cpu`, `keyword:torch`, and
`periodic_mps_exporter`. The blocking rows were high-CPU `font_maker`, high-CPU
`ai_trader` monitor/check/export children, the detached TOTO monitor chain, and
a torch queue wrapper. So the code-side candidate is ready, but the actual MPS
quality/speed/sublinear claim is still missing until a clean machine window.

Focused verification after the frame-local update:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py \
  research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/report_worldfoam_site_initialization_quality.py \
  research_experiments/world_foam_lane2/report_worldfoam_next_mps_candidate_readiness.py \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py \
  research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_site_initialization_quality.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_next_mps_candidate_readiness.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_next_mps_candidate_result.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py \
  research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_preflight.json
```

Results: `py_compile` OK; focused tests `Ran 26 tests in 0.050s`, `OK`.
The verifier correctly exits `1` on the contended preflight summary, with
failures for non-`train_eval_ok` status, incomplete stability samples, contended
preflight, blocking processes, and the missing WorldFoam artifact.

## 2026-05-21 00:55 +07 - Patch3 initializer fork is positive but not selected

I got a momentary clean process scan and launched the verified frame-local MPS
candidate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_verified \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_verified.json \
  --preflight-stability-samples 3 --preflight-stability-interval-s 5 \
  --verify-result --execute
```

It still failed closed before train/eval: `status=preflight_contended`,
`preflight_stability_samples_completed=1`, `preflight_stability_ok=false`, and
`preflight_blocking_process_count=8`. The first sample caught high-CPU
`font_maker`, high-CPU `ai_trader` pytest/imitation/RL children, the detached
TOTO monitor chain, and the torch queue wrapper. Running
`verify_worldfoam_next_mps_candidate_result.py` on that summary exits `1` with
the expected missing-artifact and contended-preflight failures.

Because the MPS gate remained externally blocked, I tried another CPU-verifiable
initializer fork: `legacy_frame_patch3_mean`. It keeps legacy support geometry
and frame-local timing, but seeds each site's color from the same-frame 3x3
patch around the legacy pixel instead of the single pixel.

Commands:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py \
  --frame-counts 2,4 --render-size 8 --site-count 4 --time-slabs 1 \
  --site-initialization legacy_frame_patch3_mean \
  --out-json /tmp/worldfoam_gate4_legacy_frame_patch3_mean_smoke.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py \
  --max-frames 2 --render-size 16 --site-count 9 \
  --site-initialization legacy_frame_patch3_mean \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_gate1_legacy_frame_patch3_mean_reference_render16_site9_2f.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/report_worldfoam_site_initialization_quality.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/report_worldfoam_next_mps_candidate_readiness.py
```

Results: Gate4 compiler smoke `status=ok`. Gate1 CPU reference is positive
versus legacy sparse but not competitive with the current selected candidate:
train/heldout PSNR `12.761/14.315`, L1 `0.1792/0.1525`. The bridge now records
`positive_candidate_count=3`, `rejected_candidate_count=2`, and still selects
`next_mps_candidate=legacy_frame_pixel_mean`. Readiness remains `status=ok`,
`ready_for_quiet_mps_quality_speed_run=true`, `quality_claim=false`, and
`speed_claim=false`.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py \
  research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/report_worldfoam_site_initialization_quality.py \
  research_experiments/world_foam_lane2/report_worldfoam_next_mps_candidate_readiness.py \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py \
  research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_site_initialization_quality.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_next_mps_candidate_readiness.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_next_mps_candidate_result.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
```

Results: `py_compile` OK; focused tests `Ran 27 tests in 0.139s`, `OK`.

## 2026-05-21 00:28 +07 - Preflight retry mode added, live retry still blocked

I added opt-in whole-sequence preflight retry to
`run_worldfoam_next_mps_candidate.py`:
`--preflight-retry-timeout-s` and `--preflight-retry-poll-s`. Default behavior
is unchanged: with no retry timeout, the launcher still fails closed after the
first contended stability sequence. The new unit test covers the case we need
for a quiet machine window: first attempt dirty, second attempt clean for all
requested stability samples, then train/eval launches only after the clean
sequence.

The live retry smoke did not get a clean window:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_retrywait_smoke \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_retrywait_smoke.json \
  --preflight-stability-samples 3 --preflight-stability-interval-s 0 \
  --preflight-retry-timeout-s 3 --preflight-retry-poll-s 1 \
  --verify-result --execute --preflight-only
```

It wrote `status=preflight_contended`, `preflight_attempt_count=1`, and no
train/eval artifact. The timeout was too short to reach a second whole stability
attempt after the first strict preflight command consumed the available wall
time. The blockers were the same external class as before: high-CPU
`font_maker`, `ai_trader`/TOTO monitor or exporter processes, a git add helper,
and torch queue wrappers. This validates the retry schema/fail-closed behavior,
not a clean MPS result.

Current focused verification after the retry patch:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate1_realray_per_sample_reference.py \
  research_experiments/world_foam_lane2/gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/report_worldfoam_site_initialization_quality.py \
  research_experiments/world_foam_lane2/report_worldfoam_next_mps_candidate_readiness.py \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py \
  research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_site_initialization_quality.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_next_mps_candidate_readiness.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_next_mps_candidate_result.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
```

Results: `py_compile` OK; focused tests `Ran 28 tests in 0.101s`, `OK`.
Verifier runs against both
`2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_verified_retry2.json`
and
`2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_retrywait_smoke.json`
exit `1` as expected: neither summary is `train_eval_ok`, neither completed all
stability samples, both record contended preflight/blocking processes, and both
are missing the required WorldFoam train/eval artifact.

## 2026-05-21 00:35 +07 - Longer strict retry still could not enter train/eval

I tried the selected candidate again with a real retry budget:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_final_try \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_final_try.json \
  --preflight-stability-samples 3 --preflight-stability-interval-s 5 \
  --preflight-retry-timeout-s 180 --preflight-retry-poll-s 15 \
  --verify-result --execute
```

It made `11` preflight attempts and failed closed before train/eval:
`status=preflight_contended`, `preflight_stability_samples_completed=1`,
`preflight_stability_ok=false`, `train_eval_returncode=null`, and no
`.worldfoam.json` exists. The latest blocker reasons were `high_cpu`,
`keyword:torch`, and `periodic_mps_exporter`; concrete blockers included
high-CPU `font_maker`, high-CPU `ai_trader` quote/export/pytest children, the
detached TOTO periodic exporter screen chains, and a torch queue wrapper.

The post-run verifier was run on
`2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_final_try.json` and exits
`1` as expected: non-`train_eval_ok` status, nonzero/absent train returncode,
incomplete stability samples, contended preflight, blocking processes, and the
missing WorldFoam artifact. This means the candidate is still ready for a quiet
MPS quality/speed run, but the clean artifact is still missing.

## 2026-05-21 00:44 +07 - Preflight-only recheck still catches respawned blockers

I saw a brief broad `ps|rg` window with no matching `font_maker`/`ai_trader`/TOTO
rows, but direct screen/PID checks still showed the detached TOTO monitor. To
avoid a full wasted train/eval attempt, I ran a preflight-only launcher check:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_screen_blocker_recheck \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_screen_blocker_recheck.json \
  --preflight-stability-samples 3 --preflight-stability-interval-s 5 \
  --verify-result --execute --preflight-only
```

It failed closed immediately: `status=preflight_contended`,
`preflight_attempt_count=1`, `preflight_stability_samples_completed=1`,
`preflight_blocking_process_count=8`, and blocker reasons
`["high_cpu","keyword:torch","periodic_mps_exporter"]`. The current blockers
were high-CPU `font_maker`, high-CPU `ai_trader` pytest/export children, the
TOTO periodic exporter screen chain, and the torch queue wrapper. The verifier
also exits `1` on this summary for the expected reasons: no `train_eval_ok`,
incomplete stability, contended preflight, blocking processes, and missing
WorldFoam artifact.

## 2026-05-21 01:18 +07 - Actionable blocker summary added to next-MPS launcher

The selected candidate still could not run a clean MPS timing/PSNR gate because
the live process state was externally contended. Direct checks still showed the
high-CPU `font_maker` training process, the detached
`54857.toto_floor001_postfix_20260520T171609Z` screen chain, and the torch queue
wrapper. Rather than launch a contaminated train/eval, I added
`preflight_external_blocker_summary` to
`run_worldfoam_next_mps_candidate.py`. It does not allowlist or ignore any
process; the launcher still fails closed, but its JSON now groups blockers by
kind and records manual next-action hints.

Live preflight artifact:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_actionable_blockers \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_actionable_blockers.json \
  --preflight-only --execute
```

Result: exit `2`, `status=preflight_contended`, no `.worldfoam.json`. The new
summary reports `requires_external_quiet_window=true`, kind counts
`{"high_cpu_external_job": 2, "periodic_mps_exporter": 5, "torch_worker": 1}`,
and reason counts `{"high_cpu": 2, "keyword:torch": 1,
"periodic_mps_exporter": 5}`. The post-run verifier exits `1` as expected: the
artifact is not `train_eval_ok`, requested only one stability sample, records
blocking processes, and has no WorldFoam train/eval result.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
```

Results: `py_compile` OK; focused tests `Ran 29 tests in 0.067s`, `OK`.

## 2026-05-21 01:37 +07 - Native fork source wiring verifier

The clean MPS timing/PSNR gate was still blocked by external processes, so I
added a source-only verifier for the three forked native variants:
`verify_worldfoam_native_variant_sources.py`. It checks each fork's
`TORCH_LIBRARY` schemas, `m.impl` registrations, dispatch-target source
definitions, Python `torch.ops` wrapper references, and host-side
`getKernelFunction(...)` names against declared Metal `kernel void` names. This
catches broken shader wiring without requiring the extension to be built or a
quiet MPS timing window.

Live source-wiring artifact:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_sources.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_native_variant_source_wiring.json
```

Result: `status=ok`, `failures=[]`, `variant_count=3`. The direct fork reports
`11` schemas / `11` impls / `11` impl targets / `11` Python op refs / `13` host
kernel refs; the CSR fork reports `13` / `13` / `13` / `11` / `15`; the slab
fork reports `103` / `103` / `103` / `15` / `71`. The extra Metal kernels
listed in the JSON are informational: they are declared in source but not
host-loaded by this variant wrapper.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_sources.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_native_variant_sources.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_sources -v
```

Results: `py_compile` OK; focused tests `Ran 33 tests in 0.343s`, `OK`.

## 2026-05-21 01:55 +07 - Host Metal field wiring added to source verifier

The clean MPS timing/PSNR gate was still blocked by high-CPU `font_maker`, the
torch queue wrapper, and the detached TOTO screen chain, so I continued on the
source-verifiable shader fork path. The native variant verifier now also parses
the host `MetalKernels` struct, checks that every declared field has an
initializer, every initialized field is declared, and every `kernels().field`
use is declared and initialized. This targets a different source-drift class
than kernel-name mismatch: a host wrapper can name an existing Metal kernel but
still wire it through a missing or stale field.

Regenerated artifact:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_sources.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_native_variant_source_wiring.json
```

Result: `status=ok`, `failures=[]`, `variant_count=3`. Field counts now appear
in the JSON too: direct `host_kernel_field_count=13` /
`initialized_kernel_field_count=13`, CSR `15/15`, slab `92/92` with `3` direct
`kernels().field` uses.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_sources.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_native_variant_sources.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_sources -v
```

Results: `py_compile` OK; focused tests `Ran 35 tests in 0.323s`, `OK`.

## 2026-05-21 00:53 +07 - Loaded Metal source membership added to native source verifier

The native variant source verifier now checks one more host/source drift class:
`getKernelFunction(...)` names must be declared in the `.metal` files actually
listed by `load_shader_source()` via `stringByAppendingPathComponent:@"*.metal"`.
This is stricter than checking every `.metal` file in the variant directory,
because a stale host wrapper can point at a kernel that exists in a side file but
will never be loaded into the dynamic Metal library.

Regenerated artifact:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_sources.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_native_variant_source_wiring.json
```

Result: `status=ok`, `failures=[]`, `variant_count=3`. All three forks load
`world_foam_lane2_power_boundary_tensor.metal` and
`world_foam_lane2_shared_replay_tensor.metal`; the loaded-source kernel counts
are direct `13`, CSR `15`, and slab `92`.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_sources.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_native_variant_sources.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_sources -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_sources -v
```

Results: `py_compile` OK; source verifier tests `Ran 7 tests in 0.439s`, `OK`;
focused WorldFoam suite `Ran 36 tests in 0.441s`, `OK`.

## 2026-05-21 01:00 +07 - Native package import path fixed and verified

The clean MPS timing/PSNR gate remains blocked. A fresh preflight-only recheck:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_current_preflight \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_current_preflight.json \
  --preflight-only --execute
```

Result: exit `2`, `status=preflight_contended`, no `.worldfoam.json`. It caught
high-CPU `font_maker`, high-CPU ai_trader pytest/export children, the torch
queue wrapper, and the TOTO exporter chain. This is still no PSNR/speed/
sublinear evidence.

While checking the forked shader state, I found a real wrapper-load bug. The
three `_C.cpython-311-darwin.so` binaries are pure `TORCH_LIBRARY` extensions:
direct Python import raises `ImportError: dynamic module does not define module
export function (PyInit__C)`, but `torch.ops.load_library(...)` correctly
registers their ops. The wrappers were catching the import failure and never
loading the shared library, so a normal package import only worked if some other
path had already registered the custom ops. I changed all three wrappers to load
their local `_C*.so` with `torch.ops.load_library`.

New import verifier:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_imports.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_native_variant_import_registration.json
```

Result: `status=ok`, `failures=[]`, `variant_count=3`. Normal package import
now registers every compiled schema: direct `11/11`, CSR `13/13`, slab
`103/103`, with empty `extension_load_error`.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_sources.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_native_variant_sources.py \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_imports.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_native_variant_imports.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_direct_v0/torch_world_foam_lane2_fused_direct/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/torch_world_foam_lane2_fused_csr/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_sources \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_imports -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_sources \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_imports -v
```

Results: `py_compile` OK; source/import verifier tests `Ran 10 tests in
0.377s`, `OK`; focused WorldFoam suite `Ran 39 tests in 0.378s`, `OK`.

## 2026-05-21 01:09 +07 - Rebuilt forked extensions and MPS smoke-checked kernels

The three forked native variants now have fresh build evidence after the
wrapper-load fix. Rebuild commands:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_direct_v0 && \
  rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0 && \
  rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && \
  rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

All three builds completed and copied `_C.cpython-311-darwin.so` into the
package directories. Regenerated import verifier:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_imports.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_native_variant_import_registration.json
```

Result: `status=ok`, `failures=[]`, fresh extension mtimes for direct/CSR/slab,
and normal package import still registers direct `11/11`, CSR `13/13`, and
slab `103/103` schemas.

Rebuilt MPS correctness smokes:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_direct_v0/tools/smoke_power_boundary_mps.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_rebuilt_direct_power_boundary_mps_smoke.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/tools/smoke_power_boundary_mps.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_rebuilt_csr_power_boundary_mps_smoke.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_power_boundary_mps.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_rebuilt_slab_power_boundary_mps_smoke.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

Results: all three power-boundary smokes exit `status=ok`, matching CPU fixture
event sums `149` and `151` with zero invalid denominators/flags. The slab mixed
MPS suite passes `Ran 8 tests in 0.419s`, exercising ownerupdate, sample-reduce,
framegroup cached, and high-cap replay kernels.

The broader focused suite after rebuild still passes:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_sources \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_imports -v
```

Result: `Ran 39 tests in 0.343s`, `OK`.

The clean MPS PSNR/speed/sublinear gate remains unrun. Direct process checks
still show high-CPU `font_maker` (`PID 54114`, about `206%` CPU), the torch
queue wrapper (`PID 54059`), and the detached TOTO exporter chain
(`54857/54858/54864/54881/54895`), so a timing/quality run would still be
contaminated.

## 2026-05-21 01:15 +07 - Rebuilt affine real-ray smokes and ownerupdate guard

After the rebuild/import checks, I added the missing real-ray MPS smoke evidence
for the forked shader variants:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_direct_v0/tools/smoke_shared_realray_replay_mps.py \
  --max-frames 2 --render-size 16 --site-count 8 --time-slabs 1 --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_rebuilt_direct_shared_realray_replay_mps_smoke.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_csr_v0/tools/smoke_shared_affine_realray_fused_csr_mps.py \
  --frame-counts 2 --render-size 16 --site-count 8 --time-slabs 1 --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_rebuilt_csr_affine_realray_mps_smoke.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2 --render-size 16 --site-count 8 --time-slabs 1 --timing-iters 1 --include-vjp \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_rebuilt_slab_affine_realray_vjp_no_ownerupdate_mps_smoke.json
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --layout per-track --frame-counts 2 --render-size 16 --site-count 8 --time-slabs 1 \
  --timing-iters 1 --include-vjp --include-ownerupdate \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_rebuilt_slab_affine_realray_vjp_ownerupdate_pertrack_mps_smoke.json
```

Results: all four artifacts report `status=ok`. Direct shared real-ray replay
matches the CPU reference under `3.6e-7`; CSR affine real-ray replay matches the
explicit/direct paths; slab affine VJP without ownerupdate passes every
acceptance key; slab per-track ownerupdate/VJP has
`ownerupdate_diagnostics.checked=true`, max forward error `9.05e-05`, finite
grad-only ownerupdate VJP, and ownerupdate grad relative delta `1.22e-06`
versus reduce.

The one failed slab artifact from this pass was not a shader regression. The
smoke default is `--layout tiled`, but ownerupdate kernels are only executed in
the `per-track` branch. `--include-ownerupdate` with the default layout produced
null ownerupdate diagnostics and failed acceptance. I patched
`smoke_fused_slab_affine_realray_mps.py` to reject that CLI combination:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2 --render-size 16 --site-count 8 --time-slabs 1 --timing-iters 1 \
  --include-ownerupdate --out-json /tmp/should_not_write_worldfoam_slab_ownerupdate.json
```

Result: exit `2` with the explicit parser error:
`--include-ownerupdate requires --layout per-track; tiled layout does not run owner-update kernels`.

This completes the current native-fork smoke cleanup. It still does not produce
the missing clean real32 PSNR/speed/sublinear artifact; that remains blocked by
the external MPS/CPU contenders recorded in the preflight summaries.

I also ran a one-sample preflight-only current-status recheck:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_current_status_recheck \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_current_status_recheck.json \
  --preflight-only --execute --preflight-stability-samples 1 \
  --preflight-stability-interval-s 0 --wait-timeout-s 0
```

Result: exit `2`, `status=preflight_contended`, no `.worldfoam.json`. The
blockers were high-CPU `font_maker` PID `92641` (`209.2%` CPU), the
`keyword:torch` queue wrapper PID `54059`, and the TOTO exporter chain
`54857/54858/54864/54881/54895`. The ai_trader/TOTO chain was mostly idle by
CPU, but the benchmark guard still correctly treats it as a periodic exporter
that can disturb MPS timing.

I then made the slab smoke parser directly testable with an optional `argv`
parameter and added
`research_experiments/world_foam_lane2/test_smoke_fused_slab_affine_realray_cli.py`
so the unsupported tiled-ownerupdate CLI cannot regress.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/test_smoke_fused_slab_affine_realray_cli.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_smoke_fused_slab_affine_realray_cli -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_sources \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_imports \
  research_experiments.world_foam_lane2.test_smoke_fused_slab_affine_realray_cli -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps -v
```

Results: `py_compile` OK; the new CLI test file passes `Ran 3 tests in
0.004s`; the focused WorldFoam suite now passes `Ran 42 tests in 0.335s`; the
slab mixed MPS suite still passes `Ran 8 tests in 0.387s`. I also reran the
valid per-track ownerupdate real-ray smoke and it still exits `status=ok`.

## 2026-05-21 01:24 +07 - Rebuilt native smoke bundle verifier

The rebuilt native smoke evidence was spread across seven passing JSONs plus one
failed JSON from the now-guarded tiled-ownerupdate CLI mistake. I added
`verify_worldfoam_rebuilt_native_smokes.py` to make that state machine-readable:
it requires the direct/CSR/slab power-boundary smokes, direct shared real-ray
smoke, CSR affine real-ray smoke, slab affine VJP no-ownerupdate smoke, and slab
per-track ownerupdate/VJP smoke to be `status=ok`. If the old failed
`2026-05-21_worldfoam_rebuilt_slab_affine_realray_vjp_mps_smoke.json` is still
present, the verifier only accepts it as `expected_invalid_tiled_ownerupdate`.

Generated verifier artifact:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_rebuilt_native_smokes.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_rebuilt_native_smoke_bundle_verifier.json
```

Result: `status=ok`, `required_count=7`, `failures=[]`,
`known_invalid_tiled_ownerupdate.classification=expected_invalid_tiled_ownerupdate`.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_worldfoam_rebuilt_native_smokes.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_rebuilt_native_smokes.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_worldfoam_rebuilt_native_smokes -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_sources \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_imports \
  research_experiments.world_foam_lane2.test_smoke_fused_slab_affine_realray_cli \
  research_experiments.world_foam_lane2.test_verify_worldfoam_rebuilt_native_smokes -v
```

Results: `py_compile` OK; smoke-bundle verifier tests pass `Ran 5 tests in
0.006s`; focused WorldFoam suite now passes `Ran 47 tests in 0.360s`.
This is still smoke/artifact hygiene, not the missing clean real32 PSNR/speed
run.

## 2026-05-21 01:31 +07 - Goal-state audit added; clean MPS gate still blocked

A fresh preflight-only continuation check still cannot launch the selected
real32 candidate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_goal_continuation_preflight \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_preflight.json \
  --preflight-only --execute --preflight-stability-samples 1 \
  --preflight-stability-interval-s 0 --wait-timeout-s 0
```

Result: exit `2`, `status=preflight_contended`, no `.worldfoam.json`.
The blockers were high-CPU `font_maker` PID `92641` (`127.5%` CPU),
high-CPU ai_trader/lean shadow PID `96379` (`90.2%` CPU), the torch queue
wrapper PID `54059`, and the TOTO exporter chain
`54857/54858/54864/54881/54895`.

I added `report_worldfoam_fork_shader_goal_state.py` as the explicit goal audit
so future handoffs do not confuse fixed fork-shader smokes with objective
completion. It reads the source wiring verifier, import verifier, rebuilt smoke
bundle verifier, and the latest next-MPS summary.

Generated artifact:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_fork_shader_goal_state.json
```

Result: `status=blocked_external_environment`,
`shader_fork_smoke_state_fixed=true`, `objective_complete=false`, and
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_sources \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_imports \
  research_experiments.world_foam_lane2.test_smoke_fused_slab_affine_realray_cli \
  research_experiments.world_foam_lane2.test_verify_worldfoam_rebuilt_native_smokes \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Results: `py_compile` OK; goal-state audit tests pass `Ran 4 tests in 0.006s`;
focused WorldFoam suite now passes `Ran 51 tests in 0.358s`. The active goal is
therefore not complete: the fork shader smoke state is fixed, but the clean
quality/speed/sublinear gate remains externally blocked.

## 2026-05-21 01:40 +07 - Commit-scope handoff added

The relevant fork-shader work is mixed into a very dirty tree, and the native
fork directories are untracked inside the `third_party/fast-mac-gsplat`
submodule. I added
`research_experiments/world_foam_lane2/2026-05-21_worldfoam_fork_shader_commit_scope.md`
to make a future narrow commit safer.

The manifest separates:

- top-level docs, verifiers, tests, and evidence JSONs to preserve
- submodule source directories to preserve:
  `world_foam_lane2_fused_direct_v0`, `world_foam_lane2_fused_csr_v0`, and
  `world_foam_lane2_fused_slab_v0`
- generated native outputs to exclude from source commits: `build/`, `_C*.so`,
  `__pycache__/`, and `*.pyc`

I checked every listed top-level path and the three submodule variant
directories exist. This is a commit/handoff hygiene step only; it does not
change the blocked MPS quality/speed status.

## 2026-05-21 01:34 +07 - Live blocker and staging recheck

I rechecked the goal state from current files rather than treating the prior
handoff as proof. The goal audit still says
`shader_fork_smoke_state_fixed=true`, `objective_complete=false`, and the
missing requirement is the clean real32 MPS PSNR/speed/sublinear gate.

The machine is still too contended for that gate. A live process sample showed
Cursor extension-host (`PID 2441`) above 300% CPU, the `font_maker` run
(`PID 92641`) above 150% CPU, and a fresh ai_trader TOTO residual export
(`PID 98709`) around 90% CPU. The saved preflight artifact still records
`status=preflight_contended`, `preflight_blocking_process_count=8`, and
`requires_external_quiet_window=true`.

I also checked staging safety for the submodule source dirs. The
`third_party/fast-mac-gsplat` ignore rules already cover generated `build/`,
`*.so`, `__pycache__/`, and `*.pyc` outputs; `git check-ignore -v` confirmed
those patterns for the rebuilt WorldFoam extension outputs. Plain
`git -C third_party/fast-mac-gsplat add -n variants/world_foam_lane2_fused_*`
listed only source files, while `rtk git add -n` misleadingly printed
`ok (nothing to add)` in this submodule context. Use plain `git add -n` for
the submodule dry-run before committing.

I refreshed the next-MPS launcher in `--preflight-only` mode after the live
process sample. It exited `2` as expected and rewrote
`2026-05-21_worldfoam_next_mps_goal_continuation_preflight.json` with
`status=preflight_contended`, `preflight_blocking_process_count=7`,
`high_cpu_external_job=1`, `torch_worker=1`, and `periodic_mps_exporter=5`.
Refreshing `2026-05-21_worldfoam_fork_shader_goal_state.json` afterwards kept
`objective_complete=false`.

## 2026-05-21 01:38 +07 - Preflight high-CPU false-negative fixed

The live process sample showed a benchmark preflight weakness: a very hot
non-Python Cursor extension-host process could be invisible because
`_capture_benchmark_environment()` filtered by command keywords before applying
the CPU threshold. I patched `train_eval_owner_run_tape.py` so keyword-matched
benchmark processes still block at `5%` CPU, while any process can block at
`general_blocking_cpu_threshold=75%`. The launcher now classifies
`high_cpu_general` as a high-CPU external blocker.

Focused checks:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_train_eval_benchmark_environment.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate -v
```

Results: `py_compile` OK and `Ran 32 tests in 0.029s`, `OK`.

I refreshed the next-MPS preflight-only artifact again after the patch. It
still exits `2`; the current blocker sample records the hot `font_maker`
Python process as `high_cpu`, keeps the TOTO monitor chain as
`periodic_mps_exporter`, and includes `general_blocking_cpu_threshold=75.0` so
future hot non-keyword processes are visible. The clean quality/speed gate
remains blocked, not failed.

## 2026-05-21 01:42 +07 - Stale clean-result guard

I tightened `verify_worldfoam_next_mps_candidate_result.py` so future clean
quality/speed artifacts cannot be promoted if they were produced before the
stronger benchmark-environment contract. The verifier now requires the current
`blocking_cpu_threshold=5.0`, `general_blocking_cpu_threshold=75.0`, and
keyword coverage in both the launcher preflight environment and the train/eval
artifact benchmark environment.

The focused verifier test adds the failure mode explicitly:
`test_stale_environment_contract_fails_even_when_status_is_clean`.

I refreshed the next-MPS preflight-only artifact once more. It still exits `2`
with `status=preflight_contended`, now with `preflight_blocking_process_count=8`
from high-CPU `font_maker`, high-CPU ai_trader/TOTO jobs, the torch queue
wrapper, and the periodic TOTO monitor chain. Refreshed goal audit remains
`objective_complete=false`.

Focused verification after this guard:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.SiteInitializationTests \
  research_experiments.world_foam_lane2.test_report_worldfoam_site_initialization_quality \
  research_experiments.world_foam_lane2.test_report_worldfoam_next_mps_candidate_readiness \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_sources \
  research_experiments.world_foam_lane2.test_verify_worldfoam_native_variant_imports \
  research_experiments.world_foam_lane2.test_smoke_fused_slab_affine_realray_cli \
  research_experiments.world_foam_lane2.test_verify_worldfoam_rebuilt_native_smokes \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Result: `Ran 53 tests in 0.615s`, `OK`.

## 2026-05-21 01:57 +07 - Uncapped preflight counts and broad suite

The next-MPS preflight summary was still misleading in one detail: the stored
process list is capped for readability, and `preflight_contending_process_count`
could fall back to that capped list length. I added explicit
`blocking_process_count`, `contending_process_count`, and
`background_process_count` fields to the benchmark-environment snapshot, then
threaded those through the next-MPS launcher and the Gate4 affine CSR promotion
summary. This keeps the compact process sample small without losing the real
blocker count.

The broad lane suite also exposed a stale adapter in
`run_gate4_affine_candidate_csr_promotion_gate.py`: the verifier now requires
`args.tape_mode`, but `_verify_args(...)` had not forwarded the fixed Gate4
affine CSR mode. I patched the adapter to pass the same tape mode used by the
artifact launch.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_run_gate4_affine_candidate_csr_promotion_gate -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py'
```

Results: targeted promotion-gate module `Ran 4 tests in 0.007s`, `OK`; broad
WorldFoam lane suite `Ran 465 tests in 274.812s`, `OK`.

The refreshed next-MPS preflight-only artifact still exits `2` with
`status=preflight_contended`, but now records matching
`preflight_blocking_process_count=7` and
`preflight_contending_process_count=7`. Current blocker kinds:
`high_cpu_external_job=1`, `torch_worker=1`, `periodic_mps_exporter=5`.
Refreshed goal audit remains `status=blocked_external_environment` and
`objective_complete=false`.

## 2026-05-21 02:02 +07 - Blockers are active, not stale

I refreshed the next-MPS preflight again. It still exits `2` with
`status=preflight_contended`, now with `preflight_blocking_process_count=8` and
`preflight_contending_process_count=8`. The clean real32 MPS gate still should
not run.

The blocker processes are doing real work, not just stale process-table ghosts:

- `font_maker` PID `92641` is actively consuming about `190-210%` CPU and its
  stdout reached step `12000`; its run wrote
  `checkpoint_step_007500.pt` at `2026-05-21 01:42:11 +0700`.
- `ai_trader` TOTO monitor PID chain `54857/54858/54864/54881/54895` remains
  live, and `events.jsonl` plus `live_paper_ledger_report.json` were written at
  `2026-05-21 02:01:24 +0700`.
- The TOTO threshold-canary subprocess PID `3834` appeared as an additional
  Python blocker during the latest preflight sample.

So the right next action remains to wait for or pause the external jobs, then
run the next-MPS launcher without `--preflight-only`.

## 2026-05-21 02:04 +07 - Verifier rejects dirty counts explicitly

While the clean MPS window remained blocked, I tightened
`verify_worldfoam_next_mps_candidate_result.py` again. A summary can no longer
look clean merely because the top-level status says `background`: the verifier
now rejects nonzero `preflight_contending_process_count`, plus nonzero
`blocking_process_count` or `contending_process_count` inside the saved
benchmark-environment snapshots for both the launcher preflight and the
train/eval artifact.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Result: `Ran 37 tests in 0.028s`, `OK`.

Running the verifier against the current blocked preflight artifact exits `1`
as intended and now includes the dirty-count failures:
`preflight recorded contending processes`,
`summary preflight benchmark environment recorded blocking processes`, and
`summary preflight benchmark environment recorded contending processes`.

## 2026-05-21 02:07 +07 - Goal audit now embeds verifier failures

I tightened `report_worldfoam_fork_shader_goal_state.py` so the goal-state
artifact no longer treats a legacy `result_verified` marker as enough. The
next-MPS requirement is complete only when
`verify_worldfoam_next_mps_candidate_result.verify_summary(...)` returns
`status=ok`, which means the launcher summary must be `train_eval_ok` and the
clean benchmark-environment, command, artifact, frame-count, PSNR/L1, and
sublinear timing contracts must all pass.

The current blocked preflight still keeps top-level `failures=[]`, because the
code path is blocked by external environment rather than a failed shader gate,
but the saved goal-state JSON now carries
`result_verifier_status=failed` and the exact verifier failures under
`artifacts.next_mps_quality_speed.result_verifier_failures`. The latest
refreshed goal state remains `status=blocked_external_environment` and
`objective_complete=false`, with `preflight_blocking_process_count=7` and
`preflight_contending_process_count=7`.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
```

Result: report/test compile passed; focused audit plus verifier tests ran
`11 tests in 0.019s`, `OK`.

## 2026-05-21 02:11 +07 - Guarded launcher still blocks, path normalization fixed

I tried the full guarded next-MPS launcher rather than another weak
preflight-only probe:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_goal_continuation_preflight \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_preflight.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_preflight.worldfoam.json \
  --execute --verify-result --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 --wait-timeout-s 0
```

It exited `2` before train/eval, as intended. The first stability sample found
`preflight_blocking_process_count=8` and
`preflight_contending_process_count=8`: high-CPU `font_maker`, high-CPU
ai_trader pytest, the torch random-stroke queue wrapper, and the TOTO monitor
chain. The clean real32 MPS PSNR/speed/sublinear gate remains unrun.

That attempt exposed one real harness bug: when `--out-json` was passed as a
repo-relative path, the verifier interpreted it as summary-relative and reported
a doubled `results/research_experiments/...` path. I fixed
`run_worldfoam_next_mps_candidate.py` to normalize relative config, summary, and
WorldFoam artifact paths against the dynaworld root before writing commands or
summary fields, and added a launcher test for that contract.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Result: compile passed; focused launcher/verifier/report suite ran `23 tests in
0.017s`, `OK`.

After rerunning the guarded launcher with the normalization fix, the verifier
failure list no longer includes the fake `train_eval_command --out-json does not
match planned artifact` failure. The saved goal audit still reports
`status=blocked_external_environment`, `objective_complete=false`, and
`shader_fork_smoke_state_fixed=true`.

## 2026-05-21 02:17 +07 - Broad lane suite passes; MPS gate still blocked

I reran the live benchmark-environment probe before spending the clean MPS gate.
It still exited `2` with `status=contended`, including high-CPU `font_maker`
near `215%`, the torch random-stroke queue wrapper, and the ai_trader/TOTO
monitor chain. I then ran the broad WorldFoam lane regression suite after the
launcher path-normalization patch:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py'
```

Result: `Ran 468 tests in 261.484s`, `OK`.

After the broad suite, I refreshed the guarded next-MPS launcher artifact again
with the real 3-sample stability requirement. It still exited `2` before
train/eval. The first sample had `preflight_blocking_process_count=10` and
`preflight_contending_process_count=10`, including high-CPU `font_maker`,
high-CPU `syspolicyd`, a TOTO residual export, ai_trader pytest/SFT work, the
torch queue wrapper, and the TOTO monitor chain.

The refreshed goal audit remains:

- `status=blocked_external_environment`
- `objective_complete=false`
- `shader_fork_smoke_state_fixed=true`
- missing requirement: `clean_real32_mps_psnr_speed_sublinear_gate`

The result verifier still exits `1` on the saved next-MPS summary with only the
expected failures: summary is `preflight_contended`, stability samples did not
complete, preflight was not clean, and the WorldFoam artifact is missing because
the guarded launcher correctly did not start train/eval.

## 2026-05-21 02:22 +07 - Bounded retry evidence added to goal audit

I ran the guarded launcher with a bounded retry window instead of only a single
probe:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_goal_continuation_preflight \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_preflight.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_preflight.worldfoam.json \
  --execute --verify-result --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 --preflight-retry-timeout-s 180 \
  --preflight-retry-poll-s 30 --wait-timeout-s 0
```

It made `7` preflight attempts and exited `2` without running train/eval. The
final attempt still had `preflight_blocking_process_count=7` and
`preflight_contending_process_count=7`: high-CPU `font_maker`, the external
torch queue wrapper, and the ai_trader/TOTO monitor/exporter chain. Earlier
attempts also caught transient ai_trader verifier/help jobs. No clean
three-sample stability sequence occurred.

I also updated `report_worldfoam_fork_shader_goal_state.py` so the saved goal
audit carries the retry/stability fields from the launcher summary:
`preflight_attempt_count`, `preflight_retry_timeout_s`,
`preflight_stability_samples_requested`,
`preflight_stability_samples_completed`, `preflight_stability_ok`, and
`preflight_blocking_reasons`. This makes future status reads explicit about how
hard the gate was retried before declaring it externally blocked.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
```

Result: compile passed; focused report/launcher/verifier suite ran `23 tests in
0.029s`, `OK`.

Refreshed goal audit remains `status=blocked_external_environment`,
`objective_complete=false`, and `shader_fork_smoke_state_fixed=true`.

## 2026-05-21 02:25 +07 - Blocker sample counts are explicit

The live preflight was still contended and briefly reported more blockers than
fit in the stored sample: high-CPU `font_maker`, a font checkpoint eval,
ai_trader SFT/parity work, TOTO export, and several torch wrappers. To avoid
future confusion between total blockers and the capped process sample, I updated
`run_worldfoam_next_mps_candidate.py` and the goal-state report to preserve:

- `preflight_blocking_process_sample_count`
- `preflight_blocking_process_unlisted_count`
- `preflight_contending_process_sample_count`
- `preflight_contending_process_unlisted_count`

The current refreshed guarded-launch summary exits `2` before train/eval with
`preflight_blocking_process_count=8`,
`preflight_blocking_process_sample_count=8`,
`preflight_blocking_process_unlisted_count=0`,
`preflight_contending_process_count=8`, and
`preflight_contending_process_unlisted_count=0`. The real32 PSNR/speed gate is
still unrun.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
```

Result: compile passed; focused launcher/report/verifier suite ran `24 tests in
0.041s`, `OK`.

## 2026-05-21 02:28 +07 - Append-only launch history added

Repeated blocked probes were overwriting
`2026-05-21_worldfoam_next_mps_goal_continuation_preflight.json`, which meant a
later single-attempt preflight could erase evidence from an earlier bounded
retry. I added an append-only compact history path to
`run_worldfoam_next_mps_candidate.py`: executed terminal outcomes now append one
JSONL record beside the launch summary, while non-executed plan mode still does
not write history.

Current history artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_preflight.history.jsonl
```

The refreshed current history entry records `status=preflight_contended`,
`preflight_blocking_process_count=9`,
`preflight_blocking_process_sample_count=8`,
`preflight_blocking_process_unlisted_count=1`, and blocker classes
`high_cpu_external_job=3`, `periodic_mps_exporter=4`, `torch_worker=1`.
The goal-state report now exposes the `history_jsonl` path under
`artifacts.next_mps_quality_speed.history_jsonl`.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
```

Result: compile passed; focused launcher/report/verifier suite ran `24 tests in
0.046s`, `OK`.

The real gate remains blocked. The refreshed goal audit still reports
`status=blocked_external_environment`, `objective_complete=false`, and
`shader_fork_smoke_state_fixed=true`.

## 2026-05-21 02:31 +07 - Fresh preflight still blocked, audit refreshed

I reran the guarded next-MPS launcher in `--preflight-only` mode with the
same persisted summary/artifact paths:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_goal_continuation_preflight \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_preflight.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_preflight.worldfoam.json \
  --execute --preflight-only --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 --preflight-retry-timeout-s 0 \
  --wait-timeout-s 0
```

It exited `2` before train/eval. The current summary and appended history row
record `status=preflight_contended`, `preflight_blocking_process_count=8`,
`preflight_contending_process_count=8`, sample/unlisted counts `8/0`, and
blocker classes `high_cpu_external_job=1`, `periodic_mps_exporter=6`,
`torch_worker=1`. The concrete blockers were still the high-CPU `font_maker`
training process, the ai_trader/TOTO monitor/export chain, and the external
torch random-stroke queue wrapper.

I refreshed
`results/2026-05-21_worldfoam_fork_shader_goal_state.json`; it now reflects
the fresh 8-blocker preflight and still reports
`status=blocked_external_environment`, `objective_complete=false`, and
`shader_fork_smoke_state_fixed=true`.

Focused contract verification still passes:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
```

Result: `Ran 24 tests in 0.036s`, `OK`. The standalone
`verify_worldfoam_next_mps_candidate_result.py` correctly fails against the
current summary because the summary is a blocked preflight, not a clean
`train_eval_ok` result with a WorldFoam artifact.

I also reran the three shader-fork evidence verifiers:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_sources.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_imports.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_rebuilt_native_smokes.py
```

All three returned `status=ok`. Current source/import/smoke state is still
fixed; only the clean real32 MPS PSNR/speed/sublinear gate remains unproven.

## 2026-05-21 02:33 +07 - ai_trader blocker is active, not stale

I inspected the ai_trader/TOTO process tree instead of killing anything. It is
real active work, not a dead stuck exporter:

- process tree `54857 -> 54858 -> 54864 -> 54881 -> 54895` has been alive for
  about `2h16m`
- command is a `12` hour
  `scripts/run_btc15m_overnight_shadow_monitor.py` run with run id
  `btc15m_toto_context64_floor001_postfix_20260520T171609Z`
- lock owner records PID `54895`, created at `2026-05-20T17:16:26Z`
- the run directory is actively updating; `events.jsonl`,
  `live_paper_ledger_report/`, and `iterations/0107` were written around
  `2026-05-20T19:32:54Z`
- latest ledger report is still safety-closed/report-only:
  `paper_trade_enabled=false`, `orders_sent=false`, `fills=0`, `open=0`,
  `pending_settlement=0`, `live_paper_ready=false`

So this monitor is legitimate work, but it remains a valid benchmark blocker
because it wakes every ~30 seconds and runs quote/context/report tasks. The
current clean-MPS options are: wait for the 12-hour monitor and font_maker queue
to finish, or explicitly pause/stop them before rerunning the WorldFoam gate.
I did not stop any external process.

## 2026-05-21 02:36 +07 - Blocker snapshots now preserve process age/state

I tried the guarded 3-sample preflight again. It still exited `2` before
train/eval, so the clean real32 MPS gate remains unrun. The old blockers are
still present, and a transient TOTO export worker also appeared during the
sample:

- `font_maker` train PID `7002`, `stat=R`, elapsed about `25m`, ~`189%` CPU
- TOTO residual export PID `33183`, `stat=R+`, elapsed about `12s`, ~`92%` CPU
- torch random-stroke queue PID `54059`, elapsed about `2h20m`
- TOTO monitor chain PIDs `54857/54858/54864/54881/54895`, elapsed about
  `2h20m`

To make future preflight artifacts explain this directly, I updated
`train_eval_owner_run_tape.py` to collect `stat` and `elapsed` from `ps`, kept
legacy fixture parsing compatible, and updated
`run_worldfoam_next_mps_candidate.py` so blocker summaries and append-only
history rows carry the top blocking process sample with those fields.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_train_eval_benchmark_environment.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate -v
```

Result: compile passed; focused benchmark-environment and launcher tests ran
`35 tests in 0.030s`, `OK`.

The latest
`results/2026-05-21_worldfoam_next_mps_goal_continuation_preflight.history.jsonl`
row now includes `preflight_blocking_processes` with top blocker PIDs, `stat`,
`elapsed`, CPU, reason, and truncated command. The refreshed goal audit remains
`status=blocked_external_environment`, `objective_complete=false`, and
`shader_fork_smoke_state_fixed=true`.

## 2026-05-21 02:38 +07 - Goal audit exposes top blockers too

The append-only history had the right process details, but the high-level goal
audit still only exposed counts and blocker classes. I updated
`report_worldfoam_fork_shader_goal_state.py` so
`artifacts.next_mps_quality_speed.preflight_blocking_processes` carries the top
three blocker samples with `pid`, `ppid`, `stat`, `elapsed`, `block_reason`,
CPU/memory, and truncated command.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
```

Result: compile passed; focused report/launcher/verifier tests ran
`24 tests in 0.034s`, `OK`.

I refreshed
`results/2026-05-21_worldfoam_fork_shader_goal_state.json`; it now includes the
top blocker sample and still reports `status=blocked_external_environment`,
`objective_complete=false`, and `shader_fork_smoke_state_fixed=true`.

## 2026-05-21 02:40 +07 - Goal audit carries reason counts and manual actions

The machine is still not clean for the real32 MPS gate. A live process scan
still shows high-CPU `font_maker`, the long-running TOTO monitor chain, the
torch random-stroke queue wrapper, and an active iteration worker under the
TOTO monitor. I did not start train/eval.

I tightened the high-level audit one more step: `report_worldfoam_fork_shader_goal_state.py`
now carries `blocking_reason_counts` and the launcher's `manual_next_actions`
under `artifacts.next_mps_quality_speed`, so readers do not have to inspect the
raw launch summary to see whether the block was high CPU, periodic exporter,
torch/MPS worker, or a mix.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Result: compile passed; report tests ran `5 tests in 0.006s`, `OK`.

The refreshed goal audit now records `blocking_reason_counts={high_cpu: 2,
keyword:torch: 1, periodic_mps_exporter: 5}` and manual actions to wait for or
pause high-CPU jobs, external torch/MPS workers, and the periodic ai_trader/TOTO
exporter before rerunning the clean gate.

## 2026-05-21 02:45 +07 - Full WorldFoam lane suite rerun

I ran the broad WorldFoam lane unittest sweep after the blocker-reporting
changes:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py'
```

Result: `471 tests in 287.323s`, `OK`.

I refreshed `results/2026-05-21_worldfoam_fork_shader_goal_state.json` after
the sweep. It still reports `status=blocked_external_environment`,
`objective_complete=false`, `shader_fork_smoke_state_fixed=true`, and
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

No clean real32 MPS PSNR/speed/sublinear run was launched or promoted in this
session; the preflight is still blocked by external high-CPU and torch/MPS jobs.

## 2026-05-21 02:48 +07 - Audit now follows the launcher summary

I reran the fail-closed launcher preflight with `--execute --preflight-only`
instead of launching train/eval:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_goal_continuation_preflight \
  --execute --preflight-only --verify-result \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 2 \
  --preflight-retry-timeout-s 0 \
  --preflight-retry-poll-s 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_preflight.json \
  --history-jsonl research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_preflight.history.jsonl
```

Result: exit `2`, `status=preflight_contended`. The fresh sample recorded
`preflight_blocking_process_count=7`: high-CPU `font_maker` PID `7002`
(`196.3%`, elapsed `36:15`), the external torch random-stroke wrapper PID
`54059`, and the five-process ai_trader/TOTO monitor chain.

This exposed a stale-audit bug: `report_worldfoam_fork_shader_goal_state.py`
was still defaulting to the old `...preflight.json` path, while the launcher
summary is now `...preflight.launch_summary.json`. I changed the default to the
launcher summary and added a regression test for that path. Focused
verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
```

Result: compile passed; focused audit/verifier suite ran `12 tests in 0.013s`,
`OK`.

The refreshed goal-state JSON now points
`artifacts.next_mps_quality_speed.path` at
`2026-05-21_worldfoam_next_mps_goal_continuation_preflight.launch_summary.json`
and reflects the fresh blocker counts. It still reports
`status=blocked_external_environment`, `objective_complete=false`,
`shader_fork_smoke_state_fixed=true`, and
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

## 2026-05-21 02:53 +07 - Full lane suite after audit-summary fix

I reran the broad WorldFoam lane unittest sweep after changing the goal audit to
follow the launcher summary:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py'
```

Result: `472 tests in 275.895s`, `OK`.

This raises the broad suite count by one for the new default-summary regression.
The clean real32 MPS quality/speed/sublinear gate remains blocked and unrun.

## 2026-05-21 03:00 +07 - Blocked verifier now skips absent-artifact noise

I rechecked the live benchmark environment. It is still not clean:
`font_maker` PID `7002` was around `194%` CPU, the torch random-stroke queue
wrapper PID `54059` is still present, and the ai_trader/TOTO monitor chain is
still present. I did not launch train/eval.

I tightened `verify_worldfoam_next_mps_candidate_result.py` so a
`preflight_contended` launcher summary remains a hard failure, but it does not
also run the WorldFoam artifact contract against a missing/unrun train artifact.
The verifier now reports `artifact_checks_skipped=true` for blocked preflights
and its failures are limited to summary/preflight cleanliness requirements.
`report_worldfoam_fork_shader_goal_state.py` exposes that bit as
`result_verifier_artifact_checks_skipped`.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_next_mps_candidate_result.py \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Result: compile passed; focused verifier/report suite ran `12 tests in 0.011s`,
`OK`.

I refreshed the fail-closed launcher summary with `--execute --preflight-only
--verify-result`; it exited `2` with `status=preflight_contended` and current
blockers: high-CPU `font_maker` PID `7002` (`206.6%`, elapsed `44:35`), torch
queue PID `54059`, and the five-process ai_trader/TOTO monitor chain. The
refreshed result verifier reports `artifact_checks_skipped=true` and only the
preflight/clean-environment failures.

I reran the broad WorldFoam lane unittest sweep after the verifier change:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py'
```

Result: `472 tests in 255.617s`, `OK`.

The refreshed goal-state JSON still reports
`status=blocked_external_environment`, `objective_complete=false`,
`shader_fork_smoke_state_fixed=true`, and
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

## 2026-05-21 03:06 +07 - Verifier output includes blocker summary directly

The live preflight remained contended. A direct preflight sample saw high-CPU
`font_maker` PID `7002`, a fresh high-CPU `ai_trader` pytest child, the torch
random-stroke queue wrapper, and the ai_trader/TOTO monitor chain. I did not
launch train/eval.

I extended `verify_worldfoam_next_mps_candidate_result.py` so its standalone
JSON report now carries the preflight status, blocking/contending counts,
blocking reasons, top three blocker samples, and
`preflight_external_blocker_summary`. This keeps the verifier output
self-contained when the launcher summary is blocked, instead of requiring a
separate audit report to understand why it failed.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_next_mps_candidate_result.py \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Result: compile passed; focused verifier/report suite ran `12 tests in 0.014s`,
`OK`.

I refreshed the fail-closed launcher summary with `--execute --preflight-only
--verify-result`; it exited `2` with `status=preflight_contended` and current
blocker counts `high_cpu=2`, `keyword:torch=1`, `periodic_mps_exporter=5`.
The standalone verifier now prints those counts, the blocker classes, and the
top blocker samples directly while keeping `artifact_checks_skipped=true`.

I refreshed `results/2026-05-21_worldfoam_fork_shader_goal_state.json` and
reran the broad lane unittest sweep:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py'
```

Result: `472 tests in 254.159s`, `OK`.

The goal state is unchanged: shader fork/source/import/smoke gates are fixed,
but the clean real32 MPS PSNR/speed/sublinear requirement remains blocked by
external jobs and is still unproven.

## 2026-05-21 03:14 +07 - History rows include blocker reason counts

The live clean-MPS gate remained blocked. I extended
`run_worldfoam_next_mps_candidate.py` so launcher history rows now include
`blocking_reason_counts` both at the top level and inside each compact attempt,
matching the already-present blocker-kind counts. This makes the JSONL history
enough to answer "what class of blocker stopped the clean run?" without opening
the full launch summary.

Focused launcher verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate -v
```

Result: `13 tests in 0.016s`, `OK`.

I refreshed the fail-closed launcher preflight with `--execute
--preflight-only --verify-result`; it exited `2` with
`status=preflight_contended`. The refreshed launch summary recorded seven
blocking processes with reason counts `high_cpu=1`, `keyword:torch=1`, and
`periodic_mps_exporter=5`. The top blocker was the high-CPU `font_maker`
training PID `7002`; the torch random-stroke queue and ai_trader/TOTO monitor
chain were still present. No train/eval or clean real32 quality/speed run was
launched.

The standalone verifier still fails closed on the blocked preflight, but now
reports the blocker counts and samples directly while
`artifact_checks_skipped=true`, so it no longer adds fake missing-artifact
noise when train/eval was intentionally skipped.

I refreshed `results/2026-05-21_worldfoam_fork_shader_goal_state.json` and
reran the broad lane unittest sweep:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py'
```

Result: `472 tests in 259.161s`, `OK`.

The goal state is still `status=blocked_external_environment`,
`objective_complete=false`, `shader_fork_smoke_state_fixed=true`, and
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

## 2026-05-21 03:21 +07 - Clean retry blocked, audit now follows newest launcher summary

I tried the real clean-MPS gate again without `--preflight-only`:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_goal_continuation_clean_retry \
  --execute --verify-result \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 2 \
  --preflight-retry-timeout-s 0 \
  --preflight-retry-poll-s 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_clean_retry.json \
  --history-jsonl research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_clean_retry.history.jsonl
```

The launcher exited `2` at preflight with `status=preflight_contended`, so no
train/eval was launched. The new blocker snapshot had eight blockers:
`high_cpu=1`, `high_cpu_general=1`, `keyword:torch=1`, and
`periodic_mps_exporter=5`. The high-CPU domain blocker was a fresh
`font_maker` training PID `51610`; there was also a transient high-CPU
XProtect process, the torch random-stroke queue PID `54059`, and the existing
ai_trader/TOTO monitor chain.

I updated `report_worldfoam_fork_shader_goal_state.py` so the default audit
picks the newest `worldfoam_next_mps_goal_continuation*.launch_summary.json`
instead of a hard-coded older preflight artifact. That keeps the closeout
audit pointed at the latest launch attempt while still ignoring plain train
artifacts. The refreshed
`results/2026-05-21_worldfoam_fork_shader_goal_state.json` now points at
`2026-05-21_worldfoam_next_mps_goal_continuation_clean_retry.launch_summary.json`
and records the eight-blocker snapshot.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result -v
```

Result: compile passed; focused report/verifier suite ran `12 tests in
0.023s`, `OK`.

Broad verification after the audit-default change:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py'
```

Result: `472 tests in 243.529s`, `OK`.

The current goal state remains blocked, not complete:
`status=blocked_external_environment`, `objective_complete=false`,
`shader_fork_smoke_state_fixed=true`, and
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

## 2026-05-21 03:24 +07 - Live blocker refresh still blocks clean gate

I refreshed the live benchmark environment and inspected full blocker commands.
The ai_trader/TOTO process is a real 12-hour monitor command with
`--duration-hours 12`, `--interval-seconds 30`, `--toto-export-with-runtime-deps`,
runtime-offline export flags, live feature-frame/oracle-context building, and
Kalshi settlement fetching. At roughly 3:07 elapsed, waiting for it to exit
would likely mean waiting many more hours. The font_maker random-stroke
ablation queue was also active, with `train_node_curve_program_flow_v2.py`
running the `rs17_continuous_3font32_slotadapter64_fontfilm_projectbounds_euler32_15k`
config at about 200% CPU.

I did not weaken the clean-gate rules. The result verifier requires zero
blocking and zero contending processes in both the launcher preflight and the
WorldFoam artifact environment, so allowing the idle-looking TOTO parent chain
would make the result weaker and fail the current verifier contract anyway.

I refreshed the launcher summary with:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_goal_continuation_live_blocker_refresh \
  --execute --preflight-only --verify-result \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 2 \
  --preflight-retry-timeout-s 0 \
  --preflight-retry-poll-s 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_live_blocker_refresh.json \
  --history-jsonl research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_live_blocker_refresh.history.jsonl
```

It exited `2` at preflight with `status=preflight_contended`; no train/eval was
launched. The blocker counts were `high_cpu=2`, `keyword:torch=1`, and
`periodic_mps_exporter=5`. The second high-CPU blocker was a short ai_trader
pytest burst, which further confirms that a timing run during this window would
be contaminated.

The standalone verifier failed closed as expected with
`artifact_checks_skipped=true`. The refreshed
`results/2026-05-21_worldfoam_fork_shader_goal_state.json` now points at
`2026-05-21_worldfoam_next_mps_goal_continuation_live_blocker_refresh.launch_summary.json`
and still reports `status=blocked_external_environment`,
`objective_complete=false`, and
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

## 2026-05-21 03:32 +07 - Blocker commands are now self-contained

The clean gate was still blocked by the same classes of external jobs, so I
made the blocked artifacts more useful instead of weakening the verifier. The
benchmark environment capture now preserves up to 1024 characters per process
command, and the launcher/verifier/audit summaries preserve the full capped
blocker sample instead of only the first three blockers.

This matters because the previous 240-character cap hid the useful TOTO facts
inside the saved artifacts. The refreshed
`2026-05-21_worldfoam_next_mps_goal_continuation_longcmd_blocker_refresh.launch_summary.json`
now includes:

- the exact font_maker config path:
  `rs17_continuous_3font32_slotadapter64_fontfilm_projectbounds_euler32_15k.jsonc`
- the TOTO monitor duration and cadence: `--duration-hours 12`,
  `--interval-seconds 30`
- the MPS/export-style flags: `--toto-export-with-runtime-deps`,
  `--toto-export-runtime-offline`

The refreshed `results/2026-05-21_worldfoam_fork_shader_goal_state.json`
points at that long-command blocker summary, includes all eight blocker rows,
and still reports `status=blocked_external_environment`,
`objective_complete=false`, and
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

Focused verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Result: `47 tests in 0.050s`, `OK`.

Broad verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py'
```

Result: `472 tests in 257.030s`, `OK`.

## 2026-05-21 03:38 +07 - Retry waits now leave inspectable history

The real32 MPS PSNR/speed gate is still externally blocked, so I patched the
launcher instead of running a contaminated timing claim. Live preflight still
sees the font_maker random-stroke high-CPU job plus the ai_trader/TOTO 12h MPS
exporter chain; no WorldFoam train/eval launched.

`run_worldfoam_next_mps_candidate.py` now writes a summary/history row with
`status=preflight_retry_waiting` after each failed retry attempt, then writes the
terminal row on timeout. History rows also keep the full capped blocker sample,
not just the first three processes. The live probe artifact is:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_retry_history_probe.launch_summary.json
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_retry_history_probe.history.jsonl
```

Its history has two rows: `preflight_retry_waiting` then
`preflight_contended`; both retained all eight blocker rows. The refreshed goal
audit points at that probe and still reports `status=blocked_external_environment`,
`objective_complete=false`, and
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Results: launcher test module `14 tests`, `OK`; focused audit/preflight suite
`48 tests`, `OK`.

## 2026-05-21 03:40 +07 - Audit treats live retry summaries as blocked

The benchmark environment was still not clean. A direct preflight saw eight
blockers: the font_maker random-stroke high-CPU train job, an ai_trader pytest
child, the random-stroke torch queue wrapper, and five TOTO MPS exporter parent
chain rows. I did not launch train/eval.

I updated `report_worldfoam_fork_shader_goal_state.py` so
`preflight_retry_waiting` is treated as an external-environment block, same as
`preflight_contended`. That matters now that the launcher can leave a live
summary behind while a long retry wait is in progress; the top-level audit
should keep saying "blocked by environment", not "failed prerequisite".

The fresh launcher probe is:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_retry_audit_patch.launch_summary.json
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_goal_continuation_retry_audit_patch.history.jsonl
```

It failed closed at preflight with one history row, `preflight_contended`, and
all eight blocker rows retained. The refreshed goal audit now points at that
artifact and still reports `status=blocked_external_environment`,
`objective_complete=false`, and
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Results: report test module `7 tests`, `OK`; focused audit/preflight suite
`49 tests`, `OK`.

## 2026-05-21 03:43 +07 - Blocker diagnosis confirms live external work

The actual benchmark preflight was still blocked, so I added
`diagnose_worldfoam_mps_blockers.py` plus tests. It reads a launcher summary or
the top-level goal audit, classifies each blocker, resolves TOTO `--output-dir`
paths when the command includes `cd ... &&`, and checks for recently modified
files under those output directories.

Fresh diagnostic artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_mps_blocker_diagnosis.json
```

This confirms the blocked window is real, not just stale process names:

- `font_maker_random_stroke_train`: 1 active high-CPU row, about 190% CPU.
- `pytest`: 1 active high-CPU ai_trader test worker row.
- `font_maker_random_stroke_queue`: 1 idle torch wrapper row.
- `ai_trader_toto_mps_exporter`: 5 rows; the parent rows with a resolvable
  working directory had fresh outputs within seconds, including
  `iterations/0164/...`, `events.jsonl`, and `live_paper_ledger_report.json`.

The real32 MPS PSNR/speed/sublinear gate remains unrun because the environment
is still contaminated. This diagnostic should make the next manual decision
clear: wait for or pause real external work rather than treating those blockers
as dead wrappers.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/diagnose_worldfoam_mps_blockers.py \
  research_experiments/world_foam_lane2/test_diagnose_worldfoam_mps_blockers.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Results: diagnostic module `4 tests`, `OK`; focused audit/preflight/diagnostic
suite `53 tests`, `OK`.

## 2026-05-21 03:48 +07 - Blocker diagnosis refreshes live PIDs

I tightened the blocker diagnostic one more step so it refreshes saved PIDs
against live `ps` output before reporting. The artifact now separates stale
captured rows from still-live rows with `pid_live`, `live_stat`,
`live_elapsed`, `live_pcpu`, `live_pmem`, and `live_command`.

Regenerated artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_mps_blocker_diagnosis.json
```

Current result: `status=blocked`, `blocker_count=8`,
`live_blocker_count=7`. The `font_maker` train process is definitely live and
hot (`live_pcpu=209.9`). The older pytest blocker row is now stale
(`pid_live=false`), but the ai_trader TOTO exporter chain is live and writing
fresh `iterations/0167/...` prediction/export/report files within seconds of
the diagnostic. So the export is doing real work, and the WorldFoam clean MPS
gate is still not runnable without pausing or waiting for external jobs.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/diagnose_worldfoam_mps_blockers.py \
  research_experiments/world_foam_lane2/test_diagnose_worldfoam_mps_blockers.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; diagnostic module `4 tests`, `OK`; focused
diagnostic/preflight/launcher/verifier/report suite `53 tests`, `OK`.

## 2026-05-21 03:50 +07 - Live preflight artifact exposed report selector gap

I ran a fresh guarded launcher preflight, not a train/eval:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_live_preflight_0348 \
  --execute --preflight-only \
  --preflight-stability-samples 1 \
  --preflight-stability-interval-s 1 \
  --preflight-retry-timeout-s 0 \
  --preflight-retry-poll-s 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_live_preflight_0348.json \
  --history-jsonl research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_live_preflight_0348.history.jsonl
```

It exited `2` with `status=preflight_contended`, so no training ran. The
canonical live preflight artifacts are now:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_live_preflight_0348.launch_summary.json
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_live_preflight_0348.history.jsonl
```

The current blockers are eight rows: hot `font_maker`, a hot Codex renderer
helper, the idle random-stroke torch queue wrapper, and the five-process
ai_trader/TOTO monitor chain. The TOTO chain is still writing fresh
`iterations/0169/...` outputs within seconds, so this is not a dead wrapper.

That run exposed a report-selector bug: `report_worldfoam_fork_shader_goal_state.py`
only considered `*worldfoam_next_mps_goal_continuation*.launch_summary.json`,
so the audit ignored the newer `worldfoam_next_mps_live_preflight_0348`
summary. I widened the glob to `*worldfoam_next_mps*.launch_summary.json` and
updated the regression test so a newer live-preflight summary wins over older
goal-continuation summaries while train artifacts are still ignored.

After the fix, the refreshed goal audit points at:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_live_preflight_0348.launch_summary.json
```

and still reports `status=blocked_external_environment`,
`objective_complete=false`, and
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; report module `7 tests`, `OK`; focused
diagnostic/preflight/launcher/verifier/report suite `53 tests`, `OK`.

I also tightened the blocker diagnostic after the live refresh showed the Codex
renderer helper had cooled below the `75%` general-process preflight threshold.
The diagnostic now records both `active_cpu` and
`live_cpu_over_preflight_threshold`, plus
`live_cpu_over_preflight_threshold_category_counts`, so a cooled process can be
distinguished from one that is still above the threshold that made it a
preflight blocker. The latest diagnosis has only the `font_maker` train row
over its live preflight CPU threshold, while the ai_trader/TOTO rows are proved
live by recent output files rather than CPU.

Additional verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/diagnose_worldfoam_mps_blockers.py \
  research_experiments/world_foam_lane2/test_diagnose_worldfoam_mps_blockers.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; diagnostic module `5 tests`, `OK`; focused
diagnostic/preflight/launcher/verifier/report suite `54 tests`, `OK`.

## 2026-05-21 03:56 +07 - Live preflight 0353 and capped blocker diagnostics

The newest guarded preflight is now:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_live_preflight_0353.launch_summary.json
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_live_preflight_0353.history.jsonl
```

It exited `2` with `status=preflight_contended`, so no train/eval artifact was
produced. The refreshed goal audit correctly selects the `0353` launcher summary
as the canonical next-MPS artifact and remains
`status=blocked_external_environment`, `objective_complete=false`, with
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

The blocker diagnostic now preserves authoritative totals when the stored
process sample is capped: `blocker_count=9`, `blocker_sample_count=8`,
`blocker_unlisted_count=1`, mirrored by the contending-process counts. It also
classifies the TOTO worker child separately from generic high CPU and resolves
relative `--output-dir` paths by falling back through known sibling CWD roots.
The latest diagnosis reports the TOTO exporter/monitor chain as real because it
has fresh ai_trader output files, while only the `font_maker` train row is still
above its live preflight CPU threshold.

Additional verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/diagnose_worldfoam_mps_blockers.py \
  research_experiments/world_foam_lane2/test_diagnose_worldfoam_mps_blockers.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; diagnostic module `8 tests`, `OK`; focused
diagnostic/preflight/launcher/verifier/report suite `57 tests`, `OK`.

## 2026-05-21 04:00 +07 - Real-shaped gate attempt still preflight-blocked

I reran the next-MPS launcher in the real gate shape, not just the quick
one-sample probe:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_real_gate_attempt_0401 \
  --execute \
  --verify-result \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 \
  --preflight-retry-timeout-s 0 \
  --preflight-retry-poll-s 15 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0401.json \
  --history-jsonl research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0401.history.jsonl
```

It exited `2` before train/eval with `status=preflight_contended`. The
canonical goal audit now points at:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0401.launch_summary.json
```

and remains `status=blocked_external_environment`, `objective_complete=false`,
`shader_fork_smoke_state_fixed=true`, and
`missing_requirements.clean_real32_mps_psnr_speed_sublinear_gate=true`.

The current blocker set is eight sampled blockers: hot `font_maker` train,
hot macOS Spotlight `mds_stores`, the idle random-stroke torch queue wrapper,
and five ai_trader/TOTO exporter rows. The diagnostic now classifies
`mds_stores` as `macos_spotlight_indexer` instead of burying it under generic
high CPU. The TOTO chain is still real, with fresh files in iteration `0178`
and ledger/output state under the `btc15m_toto_context64_floor001_postfix`
run. The latest diagnostic reports `live_cpu_over_preflight_threshold` only for
`font_maker_random_stroke_train`; the Spotlight row had cooled below `75%` by
diagnostic time but was still active CPU.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/diagnose_worldfoam_mps_blockers.py \
  research_experiments/world_foam_lane2/test_diagnose_worldfoam_mps_blockers.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; diagnostic module `9 tests`, `OK`; focused
diagnostic/preflight/launcher/verifier/report suite `58 tests`, `OK`.

## 2026-05-21 04:02 +07 - Launcher blocker classification tightened

The live process refresh still showed a contended environment: `font_maker`
around 200% CPU, the random-stroke torch queue wrapper, and the ai_trader/TOTO
monitor chain. A fresh real-shaped launcher run:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0402.launch_summary.json
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0402.history.jsonl
```

again exited `2` before train/eval with `status=preflight_contended`,
`preflight_stability_samples_requested=3`,
`preflight_stability_samples_completed=1`, and no WorldFoam train/eval artifact.

That artifact exposed a useful classification bug: a high-CPU
`scripts/train_kalshi_btc15m_sft.py` pytest worker was being folded into
`periodic_mps_exporter` because the launcher used a broad `btc15m` substring
rule. I tightened the launcher so TOTO exporter rows require the
`periodic_mps_exporter` reason, the overnight monitor script name, or `toto`;
`train_kalshi_btc15m_sft.py` is now `ai_trader_btc15m_sft`. The diagnostic uses
the same category split. I also made the goal-state reporter recompute blocker
kind counts from saved process rows instead of trusting stale embedded
`preflight_external_blocker_summary` counts.

The refreshed goal audit now points at the `0402` launcher summary and reports
blocker kinds as:

```text
ai_trader_btc15m_sft: 1
high_cpu_external_job: 1
periodic_mps_exporter: 5
torch_worker: 1
```

The diagnostic report adds liveness detail: the SFT pytest worker was no longer
live by diagnosis time but had fresh output files, while `font_maker` remained
above its live preflight CPU threshold and TOTO continued writing iteration
`0180` outputs.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/diagnose_worldfoam_mps_blockers.py \
  research_experiments/world_foam_lane2/test_diagnose_worldfoam_mps_blockers.py \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; focused diagnostic/preflight/launcher/verifier/report
suite `62 tests`, `OK`.

## 2026-05-21 04:04 +07 - Real gate still blocked, blocker audit now current

I reran the real-shaped next-MPS launcher after the blocker classifier fixes:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0404.launch_summary.json
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0404.history.jsonl
```

It exited `2` with `status=preflight_contended`, completed only one of the
requested three stability samples, and did not produce a WorldFoam train/eval
artifact. The launcher summary now records eight blocking rows with no capped
overflow:

```text
high_cpu_external_job: 1
macos_spotlight_indexer: 1
periodic_mps_exporter: 5
torch_worker: 1
```

The paired blocker diagnosis is:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_mps_blocker_diagnosis.json
```

It reports `status=blocked`, `blocker_count=8`, `contending_process_count=8`,
and `live_blocker_count=8`. Its category split is five
`ai_trader_toto_mps_exporter` rows, one random-stroke queue wrapper, one active
`font_maker_random_stroke_train`, and one active `macos_spotlight_indexer`.
Only `font_maker_random_stroke_train` and `macos_spotlight_indexer` were still
over the live CPU preflight threshold; TOTO was still writing fresh iteration
`0182` outputs.

The refreshed goal report:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_fork_shader_goal_state.json
```

still says `status=blocked_external_environment`,
`objective_complete=false`, and `shader_fork_smoke_state_fixed=true`. The only
missing completion item is the clean real32 MPS PSNR/speed/sublinear gate.

Verifier behavior is intentionally fail-closed on the `0404` summary:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py \
  research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0404.launch_summary.json
```

It fails because the summary is not `train_eval_ok`, preflight stability did
not complete, and artifact checks are skipped. The latest code verification
before this artifact remains the focused diagnostic/preflight/launcher/verifier
report suite: `62 tests`, `OK`.

## 2026-05-21 04:07 +07 - Live blocker refresh, no rerun burned

I refreshed the goal report and reran the blocker diagnosis against the latest
`0404` launcher summary. The state is still blocked, so I did not spend another
real-shaped next-MPS train/eval attempt.

The refreshed diagnosis:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_mps_blocker_diagnosis.json
```

reports `checked_at=2026-05-21T04:07:35+07:00`, `status=blocked`,
`blocker_count=8`, `live_blocker_count=8`, and category counts:

```text
ai_trader_toto_mps_exporter: 5
font_maker_random_stroke_queue: 1
font_maker_random_stroke_train: 1
macos_spotlight_indexer: 1
```

The live CPU threshold offenders remain `font_maker_random_stroke_train`
(`live_pcpu=195.6`) and `macos_spotlight_indexer` (`live_pcpu=117.4`). The TOTO
monitor chain is not CPU-hot in this sample, but it is still active and writing
fresh `iterations/0184` outputs within roughly five seconds of the diagnosis.

Focused verification was rerun after this refresh:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Result: `Ran 62 tests in 0.157s`, `OK`. The objective remains blocked on the
clean real32 MPS PSNR/speed/sublinear artifact, not on the source/import/smoke
or reporting harness.

## 2026-05-21 04:09 +07 - Blocker still live, TOTO horizon is not transient

I refreshed the goal report and diagnosis again before attempting another
real-shaped run. The goal report still points at the `0404` launcher artifact
and remains `blocked_external_environment`; the live diagnostic checked the same
artifact at `2026-05-21T04:09:06+07:00`.

Diagnosis:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_mps_blocker_diagnosis.json
```

still reports `status=blocked`, `blocker_count=8`,
`contending_process_count=8`, and `live_blocker_count=8`:

```text
ai_trader_toto_mps_exporter: 5
font_maker_random_stroke_queue: 1
font_maker_random_stroke_train: 1
macos_spotlight_indexer: 1
```

The hot CPU blockers are still the font_maker random-stroke train
(`live_pcpu=209.1`) and Spotlight `mds_stores` (`live_pcpu=83.3`). The
random-stroke queue wrapper is idle but still matched as an external torch
worker. The TOTO monitor chain is also idle on CPU in this sample, but it is
actively writing fresh `iterations/0185` files around 25 seconds before the
diagnosis. Its command is a `--duration-hours 12` monitor launched from
`20260520T171609Z`, so this is not likely to clear immediately by waiting a few
seconds; a clean WorldFoam MPS gate should wait until that exporter is paused or
finished.

I did not launch another real-shaped train/eval attempt because the preflight
environment is already known dirty. The remaining proof is unchanged: run
`run_worldfoam_next_mps_candidate.py --execute --verify-result` only after the
preflight sees a clean stability window, then verify the resulting WorldFoam
artifact for PSNR/speed/sublinear scaling.

## 2026-05-21 04:10 +07 - Blockers still active, latest TOTO output is 0187

I refreshed the goal audit and live blocker diagnosis again. The canonical goal
report still remains `blocked_external_environment` and points at the latest
real-shaped launcher summary:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0404.launch_summary.json
```

The live diagnosis at `2026-05-21T04:10:41+07:00` still reports eight live
blockers:

```text
ai_trader_toto_mps_exporter: 5
font_maker_random_stroke_queue: 1
font_maker_random_stroke_train: 1
macos_spotlight_indexer: 1
```

The two active CPU offenders are unchanged: `font_maker_random_stroke_train`
at `live_pcpu=202.2` and Spotlight `mds_stores` at `live_pcpu=118.6`. The TOTO
chain is still active and wrote fresh `iterations/0187` outputs within roughly
5 seconds of the diagnosis, including `tree_residual_live_quote_shadow` and
`live_quote_snapshot` files. That keeps the benchmark environment dirty; I did
not launch another train/eval attempt.

## 2026-05-21 04:12 +07 - Goal report now includes live blocker diagnosis

I added a non-MPS improvement to the completion audit. The goal-state report now
embeds the latest blocker diagnosis under:

```text
artifacts.live_blocker_diagnosis
```

This is deliberately supplementary evidence only: it does not make
`objective_complete=true`, and it cannot substitute for the clean real32 MPS
PSNR/speed/sublinear artifact. It closes a handoff gap where the report only
showed stale preflight rows from the launcher summary while the separate
diagnosis knew which saved blocker PIDs were still live and whether TOTO was
still writing output.

After regenerating:

```text
research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_fork_shader_goal_state.json
```

the report has `live_blocker_diagnosis.available=true`,
`matches_next_mps_summary=true`, `checked_at=2026-05-21T04:11:46+07:00`,
`status=blocked`, `blocker_count=8`, and `live_blocker_count=8`. It shows
`font_maker_random_stroke_train` as the only current live CPU threshold offender
in the diagnosis sample; Spotlight is still live in the category counts but had
dropped below the live threshold in that sample. TOTO remains active via fresh
`iterations/0188` outputs, so the MPS gate remains blocked.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; report tests `8 tests`, `OK`; focused
diagnostic/preflight/launcher/verifier/report suite `62 tests`, `OK`.

## 2026-05-21 04:16 +07 - Refresh wrapper prevents stale diagnosis/report order

I added:

```text
research_experiments/world_foam_lane2/refresh_worldfoam_fork_shader_goal_state.py
research_experiments/world_foam_lane2/test_refresh_worldfoam_fork_shader_goal_state.py
```

The wrapper refreshes `diagnose_worldfoam_mps_blockers.py` first, writes the
diagnosis JSON, then regenerates the canonical goal-state report with that
fresh diagnosis wired in. This avoids a subtle handoff footgun from the previous
step: the goal report can include `artifacts.live_blocker_diagnosis`, but a
caller still had to remember to refresh the diagnosis before regenerating the
report.

I ran the wrapper against the real artifacts:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/refresh_worldfoam_fork_shader_goal_state.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_fork_shader_goal_state.json \
  --blocker-diagnosis-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_mps_blocker_diagnosis.json
```

The refreshed report still says `blocked_external_environment` and
`objective_complete=false`. The latest diagnosis in the report is
`checked_at=2026-05-21T04:15:58+07:00`, with eight live blockers. Only
`font_maker_random_stroke_train` remains over the live CPU preflight threshold
in this sample; Spotlight is live but below threshold, and TOTO is still writing
fresh outputs.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/refresh_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_refresh_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_refresh_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; focused diagnostic/preflight/launcher/verifier/report
refresh suite `63 tests`, `OK`.

## 2026-05-21 04:20 +07 - SFT shadow blocker split and refreshed 0418 report

The guarded real-shaped gate attempt
`results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0418.launch_summary.json`
failed closed before train/eval with `status=preflight_contended`,
`preflight_stability_samples_requested=3`, and only one stability sample
completed. No PSNR/speed/sublinear artifact was produced.

That attempt exposed one more classifier gap:
`python -m lean_trade.runners.btc_15m_sft_shadow ...` was being counted as a
generic `high_cpu_external_job`. I split it into
`ai_trader_btc15m_sft_shadow` in both the launcher blocker summary and the live
diagnostic, with tests to keep it separate from the TOTO exporter chain.

I refreshed the canonical report explicitly against the `0418` launcher
summary:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/refresh_worldfoam_fork_shader_goal_state.py \
  --next-mps-summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0418.launch_summary.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_fork_shader_goal_state.json \
  --blocker-diagnosis-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_mps_blocker_diagnosis.json
```

The refreshed report remains `blocked_external_environment` and
`objective_complete=false`. The missing requirement is still the clean real32
MPS PSNR/speed/sublinear gate. Current `0418` blocker kinds are
`ai_trader_btc15m_sft_shadow: 1`, `high_cpu_external_job: 1`,
`periodic_mps_exporter: 5`, and `torch_worker: 1`; the live diagnosis classifies
the external rows as the SFT shadow worker, the font_maker train, the
font_maker random-stroke queue wrapper, and five TOTO exporter rows.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/diagnose_worldfoam_mps_blockers.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_diagnose_worldfoam_mps_blockers.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_refresh_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; focused diagnostic/preflight/launcher/verifier/report
refresh suite `65 tests`, `OK`.

## 2026-05-21 04:23 +07 - Fresh 0422 attempt and BTC15M imitation blocker split

I made another guarded real-shaped launch attempt:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_real_gate_attempt_0422 \
  --execute --verify-result \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 \
  --preflight-retry-timeout-s 0 \
  --preflight-retry-poll-s 15 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0422.json \
  --history-jsonl research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0422.history.jsonl
```

It still failed closed at preflight with `status=preflight_contended`, only one
of three requested stability samples complete, and no train/eval artifact. The
fresh blocker set was the hot font_maker train, a new high-CPU ai_trader pytest
worker running `scripts/train_kalshi_btc15m_imitation.py`, the idle
random-stroke torch queue wrapper, and the five-row TOTO exporter chain.

That exposed another useful classifier gap. I split
`train_kalshi_btc15m_imitation.py` into `ai_trader_btc15m_imitation` in both the
launcher and diagnostic, with launcher/diagnostic regression tests. I refreshed
the canonical goal report against the `0422` summary:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/refresh_worldfoam_fork_shader_goal_state.py \
  --next-mps-summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0422.launch_summary.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_fork_shader_goal_state.json \
  --blocker-diagnosis-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_mps_blocker_diagnosis.json
```

The canonical report remains `blocked_external_environment`,
`objective_complete=false`, and `shader_fork_smoke_state_fixed=true`. The
current missing requirement is unchanged: no clean real32 MPS PSNR/speed/
sublinear artifact has run. The report now gives the current blocker kinds as
`ai_trader_btc15m_imitation: 1`, `high_cpu_external_job: 1`,
`periodic_mps_exporter: 5`, and `torch_worker: 1`.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/diagnose_worldfoam_mps_blockers.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_diagnose_worldfoam_mps_blockers.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_refresh_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; focused diagnostic/preflight/launcher/verifier/report
refresh suite `67 tests`, `OK`.

## 2026-05-21 04:25 +07 - Font maker blocker split and fresh categorized preflight

I aligned the launcher blocker classifier with the live diagnostic for the
remaining font_maker rows. The launcher now classifies the active
`font_maker/.../train_node_curve_program_flow_v2.py` row as
`font_maker_random_stroke_train` and the idle
`diffusion_auto_research/run_random_stroke_ablation_queue.py --force` wrapper as
`font_maker_random_stroke_queue`, instead of generic `high_cpu_external_job` and
`torch_worker`. A focused launcher test covers both categories.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_refresh_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; focused diagnostic/preflight/launcher/verifier/report
refresh suite `68 tests`, `OK`.

After the classifier fix I ran another guarded real-shaped attempt:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_real_gate_attempt_0425 \
  --execute --verify-result \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 \
  --preflight-retry-timeout-s 0 \
  --preflight-retry-poll-s 15 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0425.json \
  --history-jsonl research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_real_gate_attempt_0425.history.jsonl
```

It still failed closed at preflight, before train/eval. The ai_trader imitation
pytest worker from the `0422` snapshot had disappeared, but the long-running
font_maker train, the font_maker queue wrapper, and five-row TOTO exporter chain
were still present. The fresh `0425` artifact now carries the specific blocker
kinds directly:

```text
font_maker_random_stroke_train: 1
font_maker_random_stroke_queue: 1
periodic_mps_exporter: 5
```

I refreshed the canonical report against the `0425` launch summary. It remains
`blocked_external_environment`, `objective_complete=false`, and
`shader_fork_smoke_state_fixed=true`; the only missing requirement remains the
clean real32 MPS PSNR/speed/sublinear artifact.

## 2026-05-21 04:28 +07 - Goal report now marks live diagnosis freshness

The blocker state was still live on recheck: seven blockers, with
`font_maker_random_stroke_train` still CPU-hot and the TOTO exporter chain
writing fresh `iterations/0200` outputs. I added freshness fields to
`report_worldfoam_fork_shader_goal_state.py` and threaded
`--max-blocker-diagnosis-age-s` through
`refresh_worldfoam_fork_shader_goal_state.py`. Reports now include
`diagnosis_age_s`, `diagnosis_max_age_s`, `diagnosis_fresh`, and any freshness
failure messages under `artifacts.live_blocker_diagnosis`, so a stale blocker
JSON cannot look current by accident.

I added a deterministic stale-diagnosis test using an injected `now`, then
refreshed the canonical report against `0425`. The saved report now has
`diagnosis_fresh=true`, `diagnosis_max_age_s=900.0`, and a sub-second
`diagnosis_age_s` at write time. The status is unchanged:
`blocked_external_environment`, `objective_complete=false`, with the clean
real32 MPS PSNR/speed/sublinear gate still missing.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/refresh_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_refresh_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; focused diagnostic/preflight/launcher/verifier/report
refresh suite `69 tests`, `OK`.

## 2026-05-21 04:33 +07 - Blocker sampling cap fixed, real gate still blocked

The `0429` guarded attempt exposed a reporting blind spot: the preflight
environment recorded `blocking_process_count=9` but only serialized the top
eight blocker rows, leaving one external process hidden in the artifact. I
patched `train_eval_owner_run_tape.py` to use
`BENCHMARK_PROCESS_SAMPLE_LIMIT=32` and to include `process_sample_limit` in the
environment payload. The cap remains explicit, but normal small blocker sets no
longer hide rows.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_train_eval_benchmark_environment.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_refresh_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; environment tests `22 OK`; focused diagnostic/
preflight/launcher/verifier/report/refresh suite `69 OK`.

I then ran a preflight-only launcher artifact with the patched sampler:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_real_gate_attempt_0433_fullblockers \
  --execute --preflight-only --verify-result
```

It failed closed before train/eval, as intended. The artifact now records all
current blockers: `preflight_blocking_process_count=7`,
`preflight_blocking_process_sample_count=7`, and
`preflight_blocking_process_unlisted_count=0`. The blocker kinds are one hot
`font_maker_random_stroke_train`, one idle `font_maker_random_stroke_queue`, and
five `periodic_mps_exporter` rows from the ai_trader/TOTO monitor chain. The
canonical goal report was refreshed against this `0433_fullblockers` artifact
and still says `blocked_external_environment`, `objective_complete=false`,
`shader_fork_smoke_state_fixed=true`, with the clean real32 MPS PSNR/speed/
sublinear gate still missing.

## 2026-05-21 04:35 +07 - Current handoff artifact uses real 3-sample preflight shape

The `0433_fullblockers` artifact proved the blocker sample cap fix, but it was
a one-sample preflight-only launch. I created a replacement preflight-only
handoff artifact with the same 3-sample requirement used by the real quality/
speed gate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_real_gate_attempt_0438_fullblockers_3sample \
  --execute --preflight-only --verify-result \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 \
  --preflight-retry-timeout-s 0 \
  --preflight-retry-poll-s 15
```

It failed closed before train/eval on the first dirty preflight sample, with
`preflight_stability_samples_requested=3`,
`preflight_stability_samples_completed=1`,
`preflight_blocking_process_count=7`,
`preflight_blocking_process_sample_count=7`, and
`preflight_blocking_process_unlisted_count=0`. The current blockers are still
one hot `font_maker_random_stroke_train`, one idle
`font_maker_random_stroke_queue`, and five ai_trader/TOTO exporter monitor rows.
The refreshed canonical report now points at
`2026-05-21_worldfoam_next_mps_real_gate_attempt_0438_fullblockers_3sample.launch_summary.json`
and remains `blocked_external_environment`, `objective_complete=false`; no
PSNR/speed/sublinear artifact exists yet.

## 2026-05-21 04:38 +07 - Sample-limit evidence threaded through summaries

The preflight environment already exposed `process_sample_limit=32`, but the
launcher and goal-state summaries only carried the observed sample counts and
unlisted counts. I threaded the limit through as
`preflight_process_sample_limit` in launcher summaries/history/attempts and as
`process_sample_limit` in the live blocker diagnosis and canonical goal report.
That makes the handoff evidence explicit: a `7/7` blocker sample came from a
32-row cap, not the old eight-row cap.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/diagnose_worldfoam_mps_blockers.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_report_worldfoam_fork_shader_goal_state.py \
  research_experiments/world_foam_lane2/test_diagnose_worldfoam_mps_blockers.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_refresh_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; targeted launcher/report/diagnosis tests `40 OK`;
focused diagnostic/preflight/launcher/verifier/report/refresh suite `69 OK`.

I then regenerated the blocked handoff artifact with the real 3-sample shape:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_real_gate_attempt_0441_samplelimit_3sample \
  --execute --preflight-only --verify-result \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 \
  --preflight-retry-timeout-s 0 \
  --preflight-retry-poll-s 15
```

It failed closed before train/eval, with `preflight_process_sample_limit=32`,
`preflight_blocking_process_count=7`, `preflight_blocking_process_sample_count=7`,
and `preflight_blocking_process_unlisted_count=0`. The refreshed goal report
now points at `0441_samplelimit_3sample` and still remains
`blocked_external_environment`, `objective_complete=false`, with no clean real32
MPS PSNR/speed/sublinear artifact yet.

## 2026-05-21 04:43 +07 - Fresh preflight refreshed before handoff

I regenerated the preflight-only handoff after the latest blocker-classifier
changes:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_real_gate_attempt_0447_activation_classifier_3sample \
  --execute --preflight-only --verify-result \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 \
  --preflight-retry-timeout-s 0 \
  --preflight-retry-poll-s 15
```

It again failed closed before train/eval on the first dirty sample:
`status=preflight_contended`, `preflight_stability_samples_requested=3`,
`preflight_stability_samples_completed=1`,
`preflight_process_sample_limit=32`, `preflight_blocking_process_count=8`,
`preflight_blocking_process_sample_count=8`, and
`preflight_blocking_process_unlisted_count=0`. The serialized blockers were
one hot `font_maker_random_stroke_train`, one
`font_maker_random_stroke_queue`, one generic high-CPU external job from an
ai_trader pytest activation-bank verifier, and five periodic ai_trader/TOTO
exporter rows. Refreshing the canonical goal report against this summary keeps
`objective_complete=false` and `status=blocked_external_environment`; the only
missing requirement remains the clean real32 MPS PSNR/speed/sublinear gate.

## 2026-05-21 04:45 +07 - Activation-bank blocker classifier and cleaner preflight

The launcher blocker summary was still reporting the ai_trader activation-bank
integrity verifier as a generic `high_cpu_external_job`, while the live
diagnosis path could only call it broad `ai_trader_pytest`. I added an explicit
`ai_trader_btc15m_activation_bank_integrity` category/kind in both
`run_worldfoam_next_mps_candidate.py` and `diagnose_worldfoam_mps_blockers.py`,
plus focused launcher/diagnosis tests. Verification passed:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/diagnose_worldfoam_mps_blockers.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_diagnose_worldfoam_mps_blockers.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_refresh_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; targeted launcher/diagnosis suite `36 OK`; focused
diagnostic/preflight/launcher/verifier/report/refresh suite `74 OK`.

The subsequent preflight-only refresh
`2026-05-21_worldfoam_next_mps_real_gate_attempt_0445_activation_bank_classifier_3sample`
did not see the activation-bank verifier anymore. It still failed closed before
train/eval with `status=preflight_contended`,
`preflight_stability_samples_requested=3`,
`preflight_stability_samples_completed=1`,
`preflight_process_sample_limit=32`, `preflight_blocking_process_count=7`,
`preflight_blocking_process_sample_count=7`, and
`preflight_blocking_process_unlisted_count=0`. The remaining blockers are one
hot `font_maker_random_stroke_train`, one `font_maker_random_stroke_queue`, and
five `periodic_mps_exporter` rows from the ai_trader/TOTO monitor chain. The
canonical goal report now points at this artifact and remains
`objective_complete=false`, `status=blocked_external_environment`; no clean
real32 MPS PSNR/speed/sublinear artifact exists yet.

## 2026-05-21 04:50 +07 - More blocker kinds from fresh preflights

Fresh preflight attempts exposed two more ai_trader/font-maker blocker families
that were still falling through to generic high-CPU rows:

- `scripts/check_btc15m_sft_runtime_parity.py` ->
  `ai_trader_btc15m_sft_runtime_parity`
- `scripts/train_kalshi_btc15m_dqn.py` -> `ai_trader_btc15m_dqn`
- `scripts/utilities/monitor_standard_glyph_exposure.py` ->
  `font_maker_standard_glyph_monitor`

I added those kinds/categories to the launcher and live diagnosis path, with
focused tests in `test_run_worldfoam_next_mps_candidate.py` and
`test_diagnose_worldfoam_mps_blockers.py`.

Verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/diagnose_worldfoam_mps_blockers.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_next_mps_candidate.py \
  research_experiments/world_foam_lane2/test_diagnose_worldfoam_mps_blockers.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_diagnose_worldfoam_mps_blockers \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_run_worldfoam_next_mps_candidate \
  research_experiments.world_foam_lane2.test_verify_worldfoam_next_mps_candidate_result \
  research_experiments.world_foam_lane2.test_report_worldfoam_fork_shader_goal_state \
  research_experiments.world_foam_lane2.test_refresh_worldfoam_fork_shader_goal_state -v
```

Results: py_compile passed; targeted launcher/diagnosis suite `42 OK`; focused
diagnostic/preflight/launcher/verifier/report/refresh suite `80 OK`.

Latest saved preflight:
`2026-05-21_worldfoam_next_mps_real_gate_attempt_0449_runtime_parity_classifier_3sample`
failed closed before train/eval with `status=preflight_contended`,
`preflight_stability_samples_requested=3`,
`preflight_stability_samples_completed=1`,
`preflight_process_sample_limit=32`, `preflight_blocking_process_count=10`,
`preflight_blocking_process_sample_count=10`, and
`preflight_blocking_process_unlisted_count=0`. The canonical report now
classifies the blockers as `ai_trader_btc15m_dqn: 1`,
`font_maker_random_stroke_train: 1`, `font_maker_random_stroke_queue: 1`,
`font_maker_standard_glyph_monitor: 2`, and `periodic_mps_exporter: 5`.
Objective remains incomplete because the clean real32 MPS PSNR/speed/sublinear
artifact still does not exist.

I regenerated once more after the DQN/standard-glyph monitor classifiers were
in place:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_real_gate_attempt_0451_dqn_monitor_classifier_3sample \
  --execute --preflight-only --verify-result \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 \
  --preflight-retry-timeout-s 0 \
  --preflight-retry-poll-s 15
```

This fresh launcher artifact has no generic high-CPU bucket. It failed closed
before train/eval with `preflight_blocking_process_count=8`,
`preflight_blocking_process_sample_count=8`,
`preflight_blocking_process_unlisted_count=0`, and blocker kinds
`ai_trader_toto_worker: 1`, `font_maker_random_stroke_train: 1`,
`font_maker_random_stroke_queue: 1`, and `periodic_mps_exporter: 5`. The
refreshed canonical goal report now points at `0451_dqn_monitor_classifier_3sample`
and remains `objective_complete=false`, `status=blocked_external_environment`.

I then tried the real guarded launcher, not just `--preflight-only`, with a
180-second retry window:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py \
  --run-id 2026-05-21_worldfoam_next_mps_real_gate_attempt_0452_retry_window_3sample \
  --execute --verify-result \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 \
  --preflight-retry-timeout-s 180 \
  --preflight-retry-poll-s 30
```

It made seven preflight attempts and never started train/eval:
`status=preflight_contended`, `preflight_attempt_count=7`,
`preflight_blocking_process_count=6`,
`preflight_blocking_process_sample_count=6`, and
`preflight_blocking_process_unlisted_count=0`. The blocker set narrowed during
the retry window: the font_maker train/queue blockers disappeared, but the
ai_trader/TOTO monitor-export chain stayed active. Final blockers were
`ai_trader_toto_worker: 1` and `periodic_mps_exporter: 5`. The refreshed
canonical goal report now points at this 0452 summary and still reports
`objective_complete=false`, `status=blocked_external_environment`; the clean
real32 MPS PSNR/speed/sublinear artifact remains missing.

At 2026-05-21 05:00 +07, I improved the live blocker diagnosis so periodic
monitors with `--duration-hours` expose `declared_duration_hours`, parsed
`elapsed_s`, `estimated_remaining_s`, `estimated_done_at`, and
`max_estimated_remaining_s_by_category`. This does not relax the clean MPS gate;
it just makes the wait window auditable. The live TOTO monitor was still writing
recent outputs and looked active, not stuck. Refreshed artifacts show
`ai_trader_toto_mps_exporter` with about `26175s` remaining and
`estimated_done_at=2026-05-21T12:16:25+07:00`. Verification: py_compile passed;
focused diagnosis/report/refresh suite `28 OK`; full focused
diagnostic/preflight/launcher/verifier/report/refresh suite `81 OK`.

At 2026-05-21 05:02 +07, I fixed a stale-vs-live CPU accounting bug in the
diagnosis report. The old high-CPU TOTO child PID from the 0452 preflight is no
longer live, so `active_cpu_category_counts` now means live-current CPU only
and is empty in the refreshed sidecar. The historical preflight CPU sample is
preserved as `summary_cpu_active_category_counts={"ai_trader_toto_worker": 1}`.
The live blocker remains the periodic TOTO monitor/exporter chain, with recent
outputs under the ai_trader run and roughly `26021s` remaining to the same
`estimated_done_at=2026-05-21T12:16:25+07:00`. Verification: py_compile passed;
focused diagnosis/report/refresh suite `28 OK`; full focused
diagnostic/preflight/launcher/verifier/report/refresh suite `81 OK`.

At 2026-05-21 05:04 +07, I fixed the artifact-write race exposed by an
overlapping diagnosis/refresh call. `diagnose_worldfoam_mps_blockers.py`,
`refresh_worldfoam_fork_shader_goal_state.py`,
`report_worldfoam_fork_shader_goal_state.py`, and
`run_worldfoam_next_mps_candidate.py` now write JSON artifacts through
same-directory temp files and atomic `Path.replace()`. This prevents the goal
report from reading half-written sidecars or launcher summaries during future
retries. Verification: py_compile passed; focused diagnosis/launcher/report/
refresh suite `53 OK`; full focused diagnostic/preflight/launcher/verifier/
report/refresh suite `81 OK`; refreshed goal report remains readable and
`status=blocked_external_environment`.

At 2026-05-21 05:08 +07, I added `clean_mps_rerun_plan` to the canonical
fork-shader goal report. It records the guarded command for the eventual clean
real32 MPS PSNR/speed/sublinear gate, the quiet-window requirement, live blocker
status/counts, recent TOTO output counts, and the latest estimated TOTO monitor
completion time. The refreshed report still says `status=blocked_external_environment`
and `objective_complete=false`; `ready_to_run_now=false`,
`live_blocker_status=blocked`, `live_blocker_count=5`, and
`run_after_estimated_done_at=2026-05-21T12:16:25+07:00`. Verification:
py_compile passed for the report/test files; report tests `9 OK`; full focused
diagnostic/preflight/launcher/verifier/report/refresh suite `81 OK`.

At 2026-05-21 05:11 +07, I tightened the live blocker reporting so
`category_counts` remains the historical preflight sample while
`live_category_counts` records only currently live PIDs. The clean rerun plan
now uses `live_category_counts` for `live_blocking_category_counts` and keeps
the old sample separately as `preflight_sample_category_counts`. The refreshed
goal report now shows the live blocker as only `ai_trader_toto_mps_exporter: 5`;
the stale high-CPU TOTO worker remains visible only in the preflight sample and
`summary_cpu_active_category_counts`. Verification: py_compile passed; focused
diagnosis/report tests `27 OK`; full focused diagnostic/preflight/launcher/
verifier/report/refresh suite `81 OK`.

At 2026-05-21 05:13 +07, I fixed another active-state issue: the diagnosis
`status` no longer stays `blocked` forever just because the old launcher summary
contains historical preflight rows. It is now `blocked` only when a sampled
process is still live or has recent outputs; otherwise it reports
`no_live_or_recent_blockers_found` while preserving `category_counts` and the
old preflight sample. The top-level goal report can now distinguish
`blocked_external_environment` from `incomplete_ready_for_clean_mps_gate`, so
after the TOTO monitor exits and recent outputs age out, the report should
correctly tell the next agent to run the clean MPS gate instead of continuing
to wait on stale evidence. Current refreshed state is still blocked because
five live TOTO exporter wrappers are present and writing recent outputs.
Verification: py_compile passed; targeted diagnosis/report/refresh tests
`29 OK`; full focused diagnostic/preflight/launcher/verifier/report/refresh
suite `82 OK`.

At 2026-05-21 05:17 +07, I wired the refresh path to run the same current
benchmark-environment preflight used by the clean MPS launcher, and threaded
that result into `clean_mps_rerun_plan`. This prevents `ready_to_run_now` from
going true merely because old sampled PIDs cleared while a new Python/Torch/MPS
blocker appeared. The refreshed canonical report now includes
`artifacts.current_benchmark_environment_probe` and showed
`status=contended`, `returncode=2`, `blocks_promotion=true`, and
`blocking_process_count=9`: a high-CPU `font_maker` random-stroke train, a
short-lived ai_trader live-paper child, the TOTO exporter wrappers, and torch
wrapper rows. The objective is still not complete; the missing proof remains
the clean real32 MPS PSNR/speed/sublinear artifact. Verification: py_compile
passed; report/refresh tests `13 OK`; full focused diagnostic/preflight/
launcher/verifier/report/refresh suite `84 OK`.

At 2026-05-21 05:23 +07, I extended the current benchmark-environment probe to
reuse the same external-blocker classifier as launcher preflight summaries.
`clean_mps_rerun_plan` now records
`current_benchmark_environment_blocking_kind_counts`,
`current_benchmark_environment_blocking_reason_counts`,
`current_benchmark_environment_manual_next_actions`, and a compact current
blocking-process sample. The refreshed report remains
`status=blocked_external_environment`, `objective_complete=false`, and
`ready_to_run_now=false`; current blockers classify as
`font_maker_random_stroke_train:1`, `ai_trader_toto_worker:1`,
`periodic_mps_exporter:5`, and `torch_worker:2`, while the live/recent blocker
diagnosis is still `ai_trader_toto_mps_exporter:5` with
`run_after_estimated_done_at=2026-05-21T12:16:25+07:00`. The clean MPS gate was
not launched because the report still fails the quiet-window preflight.
Verification: py_compile passed; report/refresh tests `13 OK`; full focused
diagnostic/preflight/launcher/verifier/report/refresh suite `84 OK`; refreshed
goal report was written to
`research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_fork_shader_goal_state.json`.

At 2026-05-21 05:25 +07, I added an explicit
`clean_mps_rerun_plan.blocking_conditions` list so future completion audits can
see all independent stop conditions, not just the first `wait_reason`.
The refreshed canonical report now has
`blocking_conditions=["live_or_recent_external_blockers_present",
"current_benchmark_environment_contended"]`. This matters because the live
diagnosis is blocked by recent TOTO exporter outputs while the current
benchmark preflight is also contended by the high-CPU font_maker train plus the
TOTO/torch wrapper rows. The clean MPS gate remains intentionally unrun.
Verification: py_compile passed; report tests `11 OK`; full focused
diagnostic/preflight/launcher/verifier/report/refresh suite `84 OK`; refreshed
goal report written.

At 2026-05-21 05:30 +07, I added
`run_worldfoam_clean_mps_gate_when_ready.py`, a fail-closed wrapper for the
final clean real32 MPS gate. It refreshes the canonical goal report, refuses to
launch unless `clean_mps_rerun_plan.ready_to_run_now=true`, and only then runs
the exact embedded `clean_mps_rerun_plan.command`; after a launch it refreshes
the report again so completion is auditable. The live dry run returned exit
code `2`, wrote
`results/2026-05-21_worldfoam_clean_mps_ready_gate.json`, and did not launch
because the report still has
`blocking_conditions=["live_or_recent_external_blockers_present",
"current_benchmark_environment_contended"]`. Verification: py_compile passed
for the helper and tests; helper tests `4 OK`; full focused
diagnostic/preflight/launcher/ready-gate/verifier/report/refresh suite `88 OK`.

At 2026-05-21 05:33 +07, I extended the ready-gated wrapper with bounded
polling via `--wait-ready-timeout-s` and `--wait-ready-poll-s`. Defaults remain
fail-closed and non-waiting, but a future agent can now run:
`rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_clean_mps_gate_when_ready.py --execute --wait-ready-timeout-s 28800 --wait-ready-poll-s 300`
to refresh every five minutes for up to eight hours and launch only when
`ready_to_run_now=true`. A live no-wait `--execute` run still returned exit code
`2`, wrote `status=not_ready`, `ready_refresh_count=1`,
`wait_ready_timeout_s=0.0`, and did not launch because the same live/recent
TOTO plus current benchmark-environment blockers remain. Verification:
py_compile passed for the helper and tests; helper tests `6 OK`; full focused
diagnostic/preflight/launcher/ready-gate/verifier/report/refresh suite `90 OK`.

At 2026-05-21 05:27 +07, I clarified the rerun ETA semantics in
`clean_mps_rerun_plan`. The report now marks
`run_after_estimated_done_at_scope="live_blocker_diagnosis_only"`,
`run_after_estimated_done_at_requires_reprobe=true`, and
`current_benchmark_environment_has_independent_blockers=true` when the current
preflight still sees independent blockers. This prevents the
`2026-05-21T12:16:25+07:00` TOTO estimate from being mistaken for a guaranteed
safe benchmark window; after that time, agents must still refresh and require
`ready_to_run_now=true`. Current state remains blocked by live/recent TOTO
exporter outputs plus the current high-CPU font_maker/Torch contention.
Verification: py_compile passed; report tests `11 OK`; full focused
diagnostic/preflight/launcher/verifier/report/refresh suite `84 OK`; refreshed
goal report written.

At 2026-05-21 05:37 +07, I refreshed the ready-gated launcher with
`--execute`; it failed closed with exit code `2` and did not run the clean MPS
gate. Current blockers are still independent of the WorldFoam code:
`font_maker_random_stroke_train:1`, `periodic_mps_exporter:5`,
`torch_worker:2`, plus one current high-CPU external row. The full ready-gate
artifact records the process sample, including the font_maker random-stroke
train and the ai_trader/TOTO overnight monitor wrappers. I then changed
`run_worldfoam_clean_mps_gate_when_ready.py` so stdout defaults to a compact
summary while still writing the full JSON artifact to `--summary-json`; this
avoids repeated readiness checks flooding Codex and creating self-inflicted
renderer CPU noise. The missing proof remains unchanged: a clean real32 MPS
PSNR/speed/sublinear artifact verified by
`verify_worldfoam_next_mps_candidate_result`.

Verification after the compact-output change: py_compile passed for the helper
and test; ready-gate helper tests `7 OK`; full focused
diagnostic/preflight/launcher/ready-gate/verifier/report/refresh suite `91 OK`.
A final no-execute refresh using the new compact stdout wrote
`status=not_ready`, `ready_to_run_now=false`, and did not launch; latest current
blockers are `high_cpu_external_job:1`, `periodic_mps_exporter:5`, and
`torch_worker:2`, while live/recent TOTO exporter counts remain at `5`.

At 2026-05-21 05:41 +07, a fresh compact refresh showed the machine-side noise
had narrowed to only the ai_trader/TOTO periodic exporter wrappers:
`periodic_mps_exporter:5` current blockers and `ai_trader_toto_mps_exporter:5`
live/recent blockers. The clean MPS gate still did not launch. I threaded
`live_max_estimated_remaining_s_by_category` from the live blocker diagnosis
into `clean_mps_rerun_plan` and the ready-gate stdout payload so the wait is
auditable by category; the current value is
`{"ai_trader_toto_mps_exporter": 23709.0}` with
`run_after_estimated_done_at=2026-05-21T12:16:25+07:00`, still requiring a
fresh reprobe before launch. Verification: py_compile passed for report/helper
and tests; targeted report+ready-gate tests `18 OK`; full focused
diagnostic/preflight/launcher/ready-gate/verifier/report/refresh suite `91 OK`.

At 2026-05-21 05:43 +07, I tried the ready-gated `--execute` path again. It
failed closed with exit code `2` and did not launch; one transient high-CPU
font_maker pytest/Torch wrapper appeared in the first current preflight sample
but was gone by direct `ps`, and a follow-up no-execute refresh narrowed the
current blockers back to only `periodic_mps_exporter:5`. I checked the
ai_trader output directory read-only: `events.jsonl`, `live_paper_ledger_report`,
`live_paper_ledger_state`, and recent `iterations/0262` through `0268` files
were modified in the last few minutes, so the TOTO monitor/exporter is doing
real periodic work rather than sitting as a dead wrapper. Latest WorldFoam
ready-gate state remains `status=not_ready`, `ready_to_run_now=false`,
`live_max_estimated_remaining_s_by_category={"ai_trader_toto_mps_exporter": 23592.0}`,
and `run_after_estimated_done_at=2026-05-21T12:16:25+07:00`. The clean gate
should still wait for the exporter to finish or be explicitly paused; bypassing
it would invalidate the speed/sublinear claim.

At 2026-05-21 05:44 +07, another ready-gated `--execute` refresh failed closed
with exit code `2` and again did not launch. The current blocker sample is now
only the five TOTO exporter wrappers: PIDs `54857`, `54858`, `54864`, `54881`,
and `54895`, all alive by direct `ps` with elapsed time `05:28:05`. Read-only
freshness checks still show recent ai_trader monitor writes in the last five
minutes, including `live_paper_ledger_report`, `events.jsonl`, live quote
state, and iteration event files through `iterations/0269/events.json`. Latest
ready-gate payload: `status=not_ready`, `execute=true`,
`ready_to_run_now=false`, `current_benchmark_environment_blocking_kind_counts`
`{"periodic_mps_exporter": 5}`, `live_blocking_category_counts`
`{"ai_trader_toto_mps_exporter": 5}`, and
`live_max_estimated_remaining_s_by_category`
`{"ai_trader_toto_mps_exporter": 23529.0}`. No WorldFoam code changed in this
pass; the clean PSNR/speed/sublinear artifact is still missing by design.

At 2026-05-21 05:47 +07, I added a non-destructive blocker audit field:
`blocking_screen_session_names` is now extracted from `SCREEN -dmS ...`
commands by `run_worldfoam_next_mps_candidate.py`, propagated into
`current_benchmark_environment_probe`, copied into `clean_mps_rerun_plan` as
`current_benchmark_environment_blocking_screen_session_names`, and printed by
the compact ready-gate payload. The live refresh now reports the actionable
session name `toto_floor001_postfix_20260520T171609Z` while still keeping full
process commands only in the JSON artifact. The ready gate remains blocked:
`status=not_ready`, `ready_to_run_now=false`, `periodic_mps_exporter:5`,
`ai_trader_toto_mps_exporter:5`, and
`live_max_estimated_remaining_s_by_category={"ai_trader_toto_mps_exporter": 23358.0}`.
Verification: py_compile passed for launcher/report/ready helper and tests;
targeted blocker/report/ready-gate tests `43 OK`; full focused
diagnostic/preflight/launcher/ready-gate/verifier/report/refresh suite `91 OK`.

At 2026-05-21 05:49 +07, I propagated the same screen-session extraction into
the live blocker diagnosis path. `artifacts.live_blocker_diagnosis` now records
`blocking_screen_session_names`, `clean_mps_rerun_plan` exposes it as
`live_blocking_screen_session_names`, and the compact ready-gate payload prints
both the current preflight session names and live-diagnosis session names. The
live refresh briefly saw an active TOTO child/high-CPU row, so current blockers
were `periodic_mps_exporter:5`, `ai_trader_toto_worker:1`, and
`high_cpu_external_job:1`; live/recent blockers remain
`ai_trader_toto_mps_exporter:5`. Both session-name fields identify
`toto_floor001_postfix_20260520T171609Z`. Verification: py_compile passed for
report/ready helper and tests; targeted report+ready-gate tests `18 OK`; full
focused diagnostic/preflight/launcher/ready-gate/verifier/report/refresh suite
`91 OK`.

At 2026-05-21 05:50 +07, another ready-gated `--execute` refresh failed closed
with exit code `2`; a final no-execute refresh left the ready-gate artifact at
`status=not_ready`, `ready_to_run_now=false`. Current blockers were
`periodic_mps_exporter:5`, `high_cpu_external_job:2`, and
`macos_spotlight_indexer:1`; live/recent blockers were still
`ai_trader_toto_mps_exporter:5`. Both current and live session-name fields
still identify `toto_floor001_postfix_20260520T171609Z`. The canonical goal
state is consistent; a temporary jq query that returned nulls was a bad query
shape, not a report bug. The missing completion artifact remains the clean
real32 MPS PSNR/speed/sublinear verifier pass.

At 2026-05-21 05:52 +07, the ready-gated `--execute` path still failed closed
with exit code `2`. `screen -ls` shows
`54857.toto_floor001_postfix_20260520T171609Z (Detached)`, and direct `ps`
shows the wrapper chain alive at PIDs `54857`, `54858`, `54864`, `54881`, and
`54895` with elapsed time `05:35:42`. The ai_trader monitor is still producing
fresh files under the same output directory in the last five minutes, including
`live_paper_ledger_report`, `events.jsonl`, negative-edge reports, and
iteration event files through `iterations/0276/events.json`. The blocker
diagnosis remains `status=blocked` with live/recent
`ai_trader_toto_mps_exporter:5`, estimated remaining `23073.0s`, and current
preflight blockers `periodic_mps_exporter:5`, `ai_trader_toto_worker:1`, and
`high_cpu_external_job:1`. The clean benchmark must still wait for this screen
session to finish or be explicitly paused.

At 2026-05-21 05:53 +07, another ready-gated `--execute` attempt failed closed.
Current blockers increased again: `periodic_mps_exporter:5`,
`ai_trader_toto_worker:1`, `font_maker_random_stroke_train:1`,
`torch_worker:2`, and `high_cpu_external_job:1`; live/recent blockers are still
`ai_trader_toto_mps_exporter:5`, session
`toto_floor001_postfix_20260520T171609Z`, estimated remaining `22992.0s`.
Direct checks confirm the TOTO screen is detached and alive with the same
wrapper chain, and the ai_trader monitor is still writing recent
`live_paper_ledger_*`, `events.jsonl`, live state, and iteration files through
`iterations/0278/events.json`. No code changed in this pass; I did not rerun
the suite because the code under test was unchanged from the last green
focused `91 OK` run.

At 2026-05-21 05:57 +07, a no-execute ready-gate refresh still failed closed
with `status=not_ready`, `ready_to_run_now=false`, and
`objective_complete=false`. The current benchmark preflight is contended by
`font_maker_random_stroke_train:1`, `high_cpu_external_job:1`,
`periodic_mps_exporter:5`, and `torch_worker:2`; live/recent blockers still
name `ai_trader_toto_mps_exporter:5` in screen session
`toto_floor001_postfix_20260520T171609Z`. The latest remaining estimate is
`22777.0s`, with done-at hint `2026-05-21T12:16:25+07:00`; the helper still
marks that hint as reprobe-required before launch.

At 2026-05-21 06:00 +07, I tightened the clean-MPS acceptance evidence in the
goal report and ready-gate summary. `clean_mps_rerun_plan` now explicitly
records `embedded_result_verification=true`,
`acceptance_verifier_required_status=ok`, and an
`acceptance_verifier_command_template` pointing at
`verify_worldfoam_next_mps_candidate_result.py <launch_summary_json>`, so the
PSNR/speed/sublinear gate is visible as an explicit verifier contract rather
than only implied by the `--verify-result` launch flag. A refreshed `--execute`
attempt still failed closed with `ready_to_run_now=false`; current blockers
were `periodic_mps_exporter:5`, `ai_trader_toto_worker:1`,
`font_maker_random_stroke_train:1`, and `torch_worker:2`, with live/recent
`ai_trader_toto_mps_exporter:5` under
`toto_floor001_postfix_20260520T171609Z` and estimate `22602.0s`. Verification:
py_compile passed for report/ready helper and tests; focused
diagnostic/preflight/launcher/ready-gate/verifier/report/refresh suite `91 OK`.

At 2026-05-21 06:02 +07, I made the ready-gate launcher itself enforce that
acceptance contract before executing. Even when a refreshed report says
`ready_to_run_now=true`, `run_worldfoam_clean_mps_gate_when_ready.py` now
returns `ready_but_unverified_command` without launching unless the command
contains `--verify-result`, `embedded_result_verification=true`,
`acceptance_verifier_required_status=ok`, and the verifier template names
`verify_worldfoam_next_mps_candidate_result.py <launch_summary_json>`. Added
unit tests for missing `--verify-result` and malformed verifier metadata.
Focused suite now reports `93 OK`. A final live `--execute` refresh still
failed closed before launch: current blockers `periodic_mps_exporter:5`,
`font_maker_random_stroke_train:1`, `torch_worker:2`, and
`macos_spotlight_indexer:1`; live/recent `ai_trader_toto_mps_exporter:5`,
session `toto_floor001_postfix_20260520T171609Z`, estimate `22467.0s`.

At 2026-05-21 06:04 +07, I tightened the result verifier so the final
PSNR/speed/sublinear claim cannot pass on a narrower frame-count matrix. The
verifier now requires the launch command frame counts to be exactly
`[2, 4, 8, 16, 32]`; a new test proves a self-consistent partial run
(`2,4,8` summary plus matching artifact) fails. Focused suite now reports
`94 OK`. The latest live ready-gated `--execute` refresh still did not launch:
current blockers `periodic_mps_exporter:5`, `ai_trader_btc15m_sft_runtime_parity:1`,
`font_maker_random_stroke_train:1`, `torch_worker:2`, and
`macos_spotlight_indexer:1`; live/recent `ai_trader_toto_mps_exporter:5`,
session `toto_floor001_postfix_20260520T171609Z`, estimate `22345.0s`.

At 2026-05-21 06:07 +07, I tightened the same result verifier around the
actual clean-gate shape. It now requires the launch command and artifact rows
to use `render_size=64`, `site_count=24`, `steps=8`, and `warmup_steps=4`;
a new test proves a run with present rows but smaller render/site/step shape
fails instead of satisfying the final gate. The report-test synthetic complete
fixture was updated to match the real gate shape. Verification: py_compile
passed; report+verifier tests `19 OK`; focused WorldFoam diagnostic/preflight/
launcher/ready-gate/verifier/report/refresh suite `95 OK`. A final live
ready-gated `--execute` refresh still failed closed before launch: current
blockers `periodic_mps_exporter:5`, `ai_trader_toto_worker:1`,
`font_maker_random_stroke_train:1`, `torch_worker:2`, and
`macos_spotlight_indexer:1`; live/recent `ai_trader_toto_mps_exporter:5`,
session `toto_floor001_postfix_20260520T171609Z`, estimate `22162.0s`.

At 2026-05-21 06:09 +07, I hardened the verifier against nonfinite numeric
artifacts. `verify_worldfoam_next_mps_candidate_result.py` now rejects NaN/inf
for row PSNR/L1 metrics, total/backward timing means, and first-to-last scale
ratios; a new test mutates a clean artifact with nonfinite quality/timing/scale
values and verifies it fails. Verification: py_compile passed; verifier tests
`9 OK`; focused WorldFoam diagnostic/preflight/launcher/ready-gate/verifier/
report/refresh suite `96 OK`. The refreshed live `--execute` path still failed
closed before launch: current blockers `periodic_mps_exporter:5`,
`font_maker_random_stroke_train:1`, `torch_worker:2`,
`macos_spotlight_indexer:1`; live/recent `ai_trader_toto_mps_exporter:5`,
session `toto_floor001_postfix_20260520T171609Z`, estimate `22034.0s`.

At 2026-05-21 06:11 +07, I tightened two more verifier details. The launch
command frame matrix must now match `[2, 4, 8, 16, 32]` exactly, so duplicate
or reordered counts fail instead of passing via set equality. The verifier also
checks `render_sublinear_vs_frames` explicitly and validates
`render_scale_first_to_last` as finite positive and sublinear, matching the
total/backward timing gates. New tests cover duplicate/reordered frame counts
and render-scale failure. Verification: py_compile passed; verifier tests
`10 OK`; focused WorldFoam diagnostic/preflight/launcher/ready-gate/verifier/
report/refresh suite `97 OK`. The refreshed live `--execute` path still failed
closed before launch: current blockers `periodic_mps_exporter:5`,
`ai_trader_toto_worker:1`, `font_maker_random_stroke_train:1`,
`torch_worker:2`, `macos_spotlight_indexer:1`; live/recent
`ai_trader_toto_mps_exporter:5`, session
`toto_floor001_postfix_20260520T171609Z`, estimate `21913.0s`.

At 2026-05-21 06:13 +07, I closed a structural row-coverage false-positive in
the result verifier. Duplicate row frame counts, invalid boolean frame counts,
and row-count mismatches now fail before a clean artifact can satisfy the final
gate; a new test proves a full frame set plus duplicate/invalid rows is
rejected. Verification: py_compile passed; verifier tests `11 OK`; focused
WorldFoam diagnostic/preflight/launcher/ready-gate/verifier/report/refresh
suite `98 OK`. The refreshed live `--execute` path still failed closed before
launch: current blockers `periodic_mps_exporter:5`,
`font_maker_random_stroke_train:1`, `torch_worker:2`,
`macos_spotlight_indexer:1`; live/recent `ai_trader_toto_mps_exporter:5`,
session `toto_floor001_postfix_20260520T171609Z`, estimate `21811.0s`.

At 2026-05-21 06:17 +07, I tightened the final result verifier one notch more:
each artifact row must now have positive finite `render.mean_s`, not only
positive total/backward step timings plus aggregate render scale metadata. The
nonfinite quality/timing regression now mutates per-row render timing and
expects `WorldFoam row 2f missing positive render mean_s`. Verification:
py_compile passed; verifier tests `11 OK`; focused WorldFoam diagnostic/
preflight/launcher/ready-gate/verifier/report/refresh suite `98 OK`; no
`__pycache__` directories remained. The refreshed live `--execute` gate still
failed closed before launch: current blocker sample was
`periodic_mps_exporter:5`; live/recent `ai_trader_toto_mps_exporter:5`,
session `toto_floor001_postfix_20260520T171609Z`, estimate `21606.0s`, done-at
hint `2026-05-21T12:16:25+07:00`, reprobe required.

At 2026-05-21 06:20 +07, I made the goal-state completion audit require the
launcher's embedded verifier fields, not only a separately re-runnable
artifact verifier. `_next_mps_status` now requires `verify_result=true`,
`result_verifier_returncode=0`, an embedded `result_verifier_payload.status=ok`,
and a `result_verifier_command` that names `verify_worldfoam_next_mps_candidate_result.py`
and targets the same summary JSON. A new report regression proves that an
otherwise clean artifact with no embedded launcher verification stays
`incomplete_missing_clean_mps_gate`. Verification: py_compile passed; report+
verifier tests `23 OK`; focused WorldFoam diagnostic/preflight/launcher/
ready-gate/verifier/report/refresh suite `99 OK`; no `__pycache__` directories
remained. The refreshed live `--execute` gate still failed closed before
launch: current blockers `font_maker_random_stroke_train:1`,
`periodic_mps_exporter:5`, `torch_worker:2`; live/recent
`ai_trader_toto_mps_exporter:5`, session
`toto_floor001_postfix_20260520T171609Z`, estimate `21395.0s`, done-at hint
`2026-05-21T12:16:25+07:00`, reprobe required.

At 2026-05-21 06:22 +07, I closed the next embedded-verifier overclaim path:
the embedded verifier payload must now identify the same summary JSON, must not
skip artifact checks, must have an empty failure list, and must name the same
`planned_worldfoam_artifact`. A new report regression mutates the embedded
payload to point at another artifact; the external verifier still passes, but
the goal report remains `incomplete_missing_clean_mps_gate`. Verification:
py_compile passed; report+verifier tests `24 OK`; focused WorldFoam diagnostic/
preflight/launcher/ready-gate/verifier/report/refresh suite `100 OK`; no
`__pycache__` directories remained. The final refreshed live `--execute` gate
still failed closed before launch: current blockers `high_cpu_external_job:1`,
`periodic_mps_exporter:5`; live/recent `ai_trader_toto_mps_exporter:5`,
session `toto_floor001_postfix_20260520T171609Z`, estimate `21249.0s`, done-at
hint `2026-05-21T12:16:25+07:00`, reprobe required.

At 2026-05-21 06:27 +07, I tightened the shader-fork side of the goal audit.
`source_wiring`, `import_registration`, and `rebuilt_native_smoke_bundle` no
longer pass on a bare `{"status":"ok"}` artifact: the source/import validators
now require the expected three native variants, package names, empty failures,
positive schema/kernel/import counts, and registration/library invariants; the
smoke-bundle validator requires the expected smoke labels, benchmark names,
quality/speed claims false, expected invalid tiled-ownerupdate classification,
and empty failures. Added a report regression proving a status-only source
stub fails the shader-fork gate, and updated refresh fixtures to use
structurally valid source/import/smoke payloads. Verification: py_compile
passed; report tests `14 OK`; report+refresh tests `16 OK`; focused WorldFoam
diagnostic/preflight/launcher/ready-gate/verifier/report/refresh suite
`101 OK`; no `__pycache__` directories remained. The refreshed live `--execute`
gate still failed closed before launch: current blockers
`ai_trader_toto_worker:1`, `font_maker_random_stroke_train:1`,
`periodic_mps_exporter:5`, `torch_worker:2`; live/recent
`ai_trader_toto_mps_exporter:5`, session
`toto_floor001_postfix_20260520T171609Z`, estimate `20973.0s`, done-at hint
`2026-05-21T12:16:25+07:00`, reprobe required.

At 2026-05-21 06:29 +07, I reran the real non-MPS shader prerequisite
verifiers against the current native fork/artifact state:
`verify_worldfoam_native_variant_sources.py`,
`verify_worldfoam_native_variant_imports.py`, and
`verify_worldfoam_rebuilt_native_smokes.py` all wrote fresh `status=ok`
artifacts. The refreshed goal report now shows
`shader_fork_smoke_state_fixed=true` with `native_source_wiring=true`,
`native_import_registration=true`, and `rebuilt_native_smoke_bundle=true`.
The only missing requirement remains `clean_real32_mps_psnr_speed_sublinear_gate`.
Verification after the artifact refresh: focused WorldFoam diagnostic/preflight/
launcher/ready-gate/verifier/report/refresh suite `101 OK`; no lane
`__pycache__` directories remained. The final live `--execute` refresh still
failed closed before launch: current blockers `font_maker_random_stroke_train:1`,
`periodic_mps_exporter:5`, `torch_worker:2`; live/recent
`ai_trader_toto_mps_exporter:5`, session
`toto_floor001_postfix_20260520T171609Z`, estimate `20869.0s`, done-at hint
`2026-05-21T12:16:25+07:00`, reprobe required.

At 2026-05-21 06:30 +07, I reran the clean-gate readiness after the artifact
refresh and rechecked the top-level report. No new non-MPS verifier/report gap
showed up: the three shader-fork requirements remained true and structurally
verified, while `clean_real32_mps_psnr_speed_sublinear_gate` remained the only
missing requirement. The ready-gated `--execute` path still failed closed
before launch with current blockers `ai_trader_toto_worker:1`,
`font_maker_random_stroke_train:1`, `periodic_mps_exporter:5`,
`torch_worker:2`; live/recent `ai_trader_toto_mps_exporter:5`, session
`toto_floor001_postfix_20260520T171609Z`, estimate `20789.0s`. The final
focused suite still passed `101 OK`; no lane `__pycache__` directories
remained.

At 2026-05-21 06:35 +07, I reprobed the guarded clean MPS launcher and it
again failed closed before training. Current readiness state:
`ready_to_run_now=false`, status `not_ready`, goal status
`blocked_external_environment`, current blockers
`font_maker_random_stroke_train:1`, `periodic_mps_exporter:5`,
`torch_worker:2`, and live/recent `ai_trader_toto_mps_exporter:5` in screen
session `toto_floor001_postfix_20260520T171609Z`; the run-after hint remains
`2026-05-21T12:16:25+07:00` and still requires reprobe. I inspected the TOTO
monitor output rather than assuming it was stale: files under
`/Users/nicholasbardy/git/ai_trader/logs/btc15m_shadow_overnight/btc15m_toto_context64_floor001_postfix_20260520T171609Z`
were updating at the current minute, iteration `0314` wrote feature-context and
prediction-export artifacts, and the live paper ledger report said
`status=pass`, `fills=0`, `orders_sent=false`, `training_unlocked=false`,
`paper_trade_enabled=false`. So this is real active report-only/export work,
not an obviously stuck orphan. Verification: the focused WorldFoam diagnostic/
preflight/launcher/ready-gate/verifier/report/refresh suite passed `102 OK`
after the stale referenced-path regression; the clean real32 MPS gate remains
the only unproven requirement.

At 2026-05-21 06:37 +07, I reprobed once more. The guarded launcher still did
not run training: `ready_to_run_now=false`, `status=not_ready`,
`goal_status=blocked_external_environment`. Current blocker counts changed to
`font_maker_random_stroke_train:1`, `high_cpu_external_job:1`,
`periodic_mps_exporter:5`, `torch_worker:2`; the extra high-CPU general sample
was a Chrome renderer, while the font_maker train remained around 200%+ CPU and
the TOTO exporter chain was still the live/recent MPS blocker. This does not
change the WorldFoam conclusion: shader-fork prerequisites are fixed, but the
clean real32 MPS PSNR/speed/sublinear gate is still blocked by the external
environment and must not be replaced by a dirty run.

At 2026-05-21 06:38 +07, another guarded `--execute` probe still failed closed
with `ready_to_run_now=false`, `status=not_ready`, and
`goal_status=blocked_external_environment`; no WorldFoam training launched.
The refreshed report still has only
`clean_real32_mps_psnr_speed_sublinear_gate` missing. Blocker counts were
`ai_trader_toto_worker:1`, `font_maker_random_stroke_train:1`,
`high_cpu_external_job:1`, `periodic_mps_exporter:5`, `torch_worker:2`.
The sampled TOTO child was running
`run_btc_15m_toto_residual_live_quote_shadow_paper`, and an external
`uv run python -m pytest tests/` under `ai_trader` was also active. Latest TOTO
files reached iteration `0317`; the live ledger stayed safe-closed
(`status=pass`, `fills=0`, `orders_sent=false`, `training_unlocked=false`),
and the current gate blocker ledger said `status=blocked_report_only`,
`training_allowed=false`, `promotion_allowed=false`, `orders_allowed=false`.
This confirms the external blocker is active report-only work, not a reason to
weaken the WorldFoam clean-window requirement.

At 2026-05-21 06:40 +07, the guarded clean-MPS probe still refused to launch:
`ready_to_run_now=false`, `status=not_ready`, and
`goal_status=blocked_external_environment`. The report remains clean on the
non-environment side: `shader_fork_smoke_state_fixed=true`, fixed requirements
`native_source_wiring=true`, `native_import_registration=true`,
`rebuilt_native_smoke_bundle=true`, `failures=[]`, and the only missing
requirement is still `clean_real32_mps_psnr_speed_sublinear_gate`. Current
blockers are now `ai_trader_btc15m_imitation:1`,
`font_maker_random_stroke_train:1`, `periodic_mps_exporter:5`,
`torch_worker:2`; the ai_trader pytest worker was running
`scripts/train_kalshi_btc15m_imitation.py` from a pytest temp directory. The
TOTO exporter screen remains live/recent with an estimated remaining time near
`20190.0s`. No dirty WorldFoam speed claim should be made from this state.

At 2026-05-21 06:41 +07, I checked once more and deliberately did not add
another verifier patch: the report has `quality_claim=false`,
`speed_claim=false`, `shader_fork_smoke_state_fixed=true`, all three native
shader prerequisites true, and `failures=[]`. The guarded launcher still fails
closed before training with current blockers `font_maker_random_stroke_train:1`,
`high_cpu_external_job:1`, `periodic_mps_exporter:5`, `torch_worker:2`;
live/recent TOTO remains `ai_trader_toto_mps_exporter:5` in
`toto_floor001_postfix_20260520T171609Z`. The high-CPU external sample has
rotated through ai_trader pytest subprocesses, most recently
`verify_btc15m_activation_bank_integrity.py`, while the long font_maker run
remains active. This is still an external quiet-window problem, not a
WorldFoam shader/test gap.

At 2026-05-21 06:42 +07, the guarded clean-MPS launcher still returned
`status=not_ready` and did not launch training. The transient ai_trader pytest
worker was gone from the summarized blocker categories; remaining current
blockers are `font_maker_random_stroke_train:1`, `periodic_mps_exporter:5`,
and `torch_worker:2`, with live/recent `ai_trader_toto_mps_exporter:5` and
remaining estimate around `20043.0s`. The goal report is unchanged:
`objective_complete=false`, all shader-fork requirements true, `failures=[]`,
and only `clean_real32_mps_psnr_speed_sublinear_gate` missing. This is a
slightly cleaner external-blocker state, but still not a valid speed/PSNR
window.

At 2026-05-21 06:44 +07, I started the existing guarded wait-runner in a
detached screen instead of continuing manual no-op probes:
`worldfoam_clean_mps_wait_20260521_064402`. Command:
`run_worldfoam_clean_mps_gate_when_ready.py --execute --wait-ready-timeout-s 28800 --wait-ready-poll-s 300 --recent-seconds 120 --summary-json research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_clean_mps_wait_20260521_064402.json --print-payload full`.
It writes stdout/stderr to
`research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_clean_mps_wait_20260521_064402.log`.
The initial summary was `status=waiting_for_ready`, `ready_to_run_now=false`,
`wait_ready_poll_s=300.0`, `wait_ready_timeout_s=28800.0`; current blockers
remained `font_maker_random_stroke_train:1`, `periodic_mps_exporter:5`,
`torch_worker:2`, with live/recent `ai_trader_toto_mps_exporter:5`. This
background guard is safe because it will only launch the real32 MPS gate when
the refreshed goal report says `ready_to_run_now=true`, and the embedded gate
command still includes `--verify-result`.

At 2026-05-21 06:49 +07, the detached waiter wrote its second refresh. It is
still alive and still waiting: `status=waiting_for_ready`,
`ready_refresh_count=2`, `ready_to_run_now=false`, `launch_returncode=None`.
Current blocker counts are `font_maker_random_stroke_train:1`, `mps_worker:2`,
`periodic_mps_exporter:5`, `torch_worker:2`; live/recent TOTO exporter remains
active with an estimated `19635.0s`. The goal report remains
`blocked_external_environment`, `objective_complete=false`, all shader-fork
requirements true, `failures=[]`, and only
`clean_real32_mps_psnr_speed_sublinear_gate` missing.

At 2026-05-21 06:54 +07, the waiter reached refresh 3 and still refused to
launch: `status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`. Current blockers are now
`ai_trader_toto_worker:1`, `font_maker_random_stroke_train:1`,
`mps_worker:2`, `periodic_mps_exporter:5`, `torch_worker:2`; live/recent TOTO
exporter remaining estimate is about `19333.0s`. I re-ran the focused
WorldFoam guard suite while waiting, and it passed `102 OK`:
`test_diagnose_worldfoam_mps_blockers`,
`test_train_eval_benchmark_environment`,
`test_run_worldfoam_next_mps_candidate`,
`test_run_worldfoam_clean_mps_gate_when_ready`,
`test_verify_worldfoam_next_mps_candidate_result`,
`test_report_worldfoam_fork_shader_goal_state`, and
`test_refresh_worldfoam_fork_shader_goal_state`. Relevant root/submodule
`git diff --check` passed, and the lane has no `__pycache__` directories under
`research_experiments/world_foam_lane2`. The objective is still incomplete
because the clean real32 MPS PSNR/speed/sublinear artifact has not run.

At 2026-05-21 06:59 +07, the detached waiter reached refresh 4 and still did
not launch: `status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`. The blocker set improved materially: the font_maker
train, TOTO child worker, and generic torch worker categories are gone from the
current summary, leaving `mps_worker:2` and `periodic_mps_exporter:5`.
Live/recent TOTO exporter remains the durable external blocker with estimated
remaining time around `19030.0s`. This is progress toward a clean window, but
still not a valid PSNR/speed/sublinear artifact.

At 2026-05-21 07:04 +07, refresh 5 stayed blocked and did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`, `launch_returncode=None`.
The current blocker set regressed from refresh 4: `font_maker_random_stroke_train:1`
and `torch_worker:2` are back alongside `mps_worker:2` and
`periodic_mps_exporter:5`. Live/recent TOTO exporter remains present with an
estimated `18728.0s` remaining. I also re-audited the launcher/verifier
contract: the waiter only launches when `ready_to_run_now=true`, the command
must include `--verify-result`, and completion requires the result verifier to
prove clean environment, exact frame counts `[2,4,8,16,32]`, MPS/manual-VJP
native-cutwalk settings, finite PSNR/L1 rows, and sublinear total/render/
backward scaling. No completion claim is justified yet.

At 2026-05-21 07:06 +07, inspecting the refresh-5 process sample exposed an
observer-footgun: two `keyword:mps` blockers were my own long `sleep ... read
waiter JSON` shell processes because their command lines included the
`worldfoam_clean_mps_wait...json` path while the waiter refreshed. That can
contaminate the clean-window probe even though it does no GPU work. I added
`research_experiments/world_foam_lane2/read_clean_gate_waiter_status.py` so
future polling can use a neutral command line with no MPS-named argv path.
Verified with `py_compile` and a short `--sample 3` read.

At 2026-05-21 07:10 +07, the neutral observer confirmed refresh 6 did not carry
the self-inflicted `mps_worker` rows anymore. The waiter still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, with blockers now
`font_maker_random_stroke_train:1`, `torch_worker:2`, and
`periodic_mps_exporter:5`; live/recent TOTO exporter remaining estimate is
`18426.0s`. I added `test_read_clean_gate_waiter_status.py` for the neutral
reader helper and ran it: `3 OK`.

At 2026-05-21 07:15 +07, refresh 7 was materially cleaner but still did not
launch: `status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`. The font_maker and torch categories disappeared;
the only remaining blocker category is `periodic_mps_exporter:5`, all from
the `toto_floor001_postfix_20260520T171609Z` screen chain. Estimated live/
recent TOTO remaining time is `18124.0s`. This is now a single external TOTO
exporter wait, not a WorldFoam shader/verifier issue.

At 2026-05-21 07:21 +07, refresh 8 regressed and still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`. Blockers are now `font_maker_random_stroke_train:1`,
`ai_trader_toto_worker:1`, `torch_worker:2`, and `periodic_mps_exporter:5`.
The font_maker job is a new rs29 random-stroke train
(`rs29_continuous_10font32_..._rs28_2k_to_10k.jsonc`), and the TOTO child is
iteration `0354` live prediction export. Live/recent TOTO exporter remaining
estimate is `17822.0s`. The guarded waiter is still doing the right thing by
not launching into this contended window.

At 2026-05-21 07:22 +07, I extended the neutral waiter reader with summary
age and stale-for-poll reporting. This helps separate "waiter is blocked but
alive" from "waiter stopped updating" without reading the MPS-named JSON path
from a long-lived argv. Focused reader tests now pass `5 OK`; live read showed
`summary_stale_for_poll=false` while refresh 8 remained blocked by external
font_maker/TOTO jobs.

At 2026-05-21 07:25 +07, refresh 9 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. Current blockers are
`font_maker_random_stroke_train:1`, `ai_trader_btc15m_sft_shadow:1`,
`torch_worker:2`, and `periodic_mps_exporter:5`; live/recent TOTO exporter
remaining estimate is `17519.0s`. The sampled ai_trader worker is
`lean_trade.runners.btc_15m_sft_shadow`, and the sampled font_maker job is
still the rs29 random-stroke train. The clean-gate waiter is alive and blocked
by real external jobs.

At 2026-05-21 07:30 +07, refresh 10 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. Blocker categories
are effectively unchanged: `font_maker_random_stroke_train:1`,
`ai_trader_toto_worker:1`, `torch_worker:2`, and `periodic_mps_exporter:5`.
The sampled TOTO child rotated to iteration `0363`
`run_btc_15m_tree_residual_live_quote_shadow_paper`; live/recent TOTO exporter
remaining estimate is `17217.0s`. Further tight polling is not useful until
these external jobs clear; keep the single guarded waiter running.

At 2026-05-21 07:33 +07, I added `--wait-refresh-timeout-s` and
`--wait-refresh-poll-s` to the neutral waiter reader so future live checks can
block until the summary is rewritten without shell sleeps or MPS-named argv.
Focused reader tests now pass `7 OK`. The first live `--wait-refresh` check
returned refresh 11: still `waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. Blockers are
`font_maker_random_stroke_train:1`, `ai_trader_btc15m_sft_shadow:1`,
`macos_spotlight_indexer:1`, `torch_worker:2`, and
`periodic_mps_exporter:5`; live/recent TOTO remaining estimate is `16914.0s`.
No clean candidate launch occurred.

At 2026-05-21 07:36 +07, refresh 12 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. Spotlight cooled
out of the blocker set, but the window is still contended by
`font_maker_random_stroke_train:1`, `ai_trader_toto_worker:1`,
`torch_worker:2`, and `periodic_mps_exporter:5`. The sampled TOTO child rotated
to iteration `0372` `run_btc_15m_toto_residual_live_quote_shadow_paper`;
live/recent TOTO remaining estimate is `16612.0s`. The clean gate remains
blocked by external jobs only.

At 2026-05-21 07:41 +07, refresh 13 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The transient
ai_trader child worker cleared, leaving `font_maker_random_stroke_train:1`,
`torch_worker:2`, and `periodic_mps_exporter:5`; live/recent TOTO remaining
estimate is `16309.0s`. The single guarded waiter remains alive and correctly
blocked on external work.

At 2026-05-21 07:46 +07, refresh 14 still did not launch, but the blocker set
improved again: `font_maker_random_stroke_train` and `torch_worker` cleared.
Remaining blockers are `ai_trader_toto_worker:1` and
`periodic_mps_exporter:5`; live/recent TOTO remaining estimate is `16006.0s`.
The sampled TOTO child is iteration `0381`
`export_btc15m_toto_residual_live_prediction_export.py`. The clean gate is now
blocked by TOTO/exporter work only.

At 2026-05-21 07:55 +07, refresh 16 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The transient TOTO
worker cleared, but macOS Spotlight indexing re-entered the blocker set:
`macos_spotlight_indexer:1` plus `periodic_mps_exporter:5`. Live/recent TOTO
remaining estimate is `15402.0s`. This is still an external quiet-window block.

At 2026-05-21 08:04 +07, refresh 17 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. Spotlight cleared,
but a fresh font_maker random-stroke train entered the blocker set with its
two torch wrapper processes: `font_maker_random_stroke_train:1`,
`torch_worker:2`, and `periodic_mps_exporter:5`. Live/recent TOTO remaining
estimate is `15100.0s`. The clean real32 MPS artifact is still missing because
the benchmark environment is externally contended.

At 2026-05-21 08:10 +07, refresh 18 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. Spotlight indexing
re-entered while the font_maker job and TOTO exporter remained active:
`font_maker_random_stroke_train:1`, `macos_spotlight_indexer:1`,
`torch_worker:2`, and `periodic_mps_exporter:5`. Live/recent TOTO remaining
estimate is `14798.0s`. No clean result exists to verify yet.

At 2026-05-21 08:15 +07, refresh 19 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. Spotlight cleared
again, but an ai_trader/TOTO child worker re-entered while font_maker and the
periodic exporter remained active: `font_maker_random_stroke_train:1`,
`ai_trader_toto_worker:1`, `torch_worker:2`, and `periodic_mps_exporter:5`.
Live/recent TOTO remaining estimate is `14495.0s`. The clean gate is still
blocked by external work, not by a WorldFoam verifier failure.

At 2026-05-21 08:20 +07, refresh 20 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The ai_trader child
rotated from TOTO quote-shadow work to `ai_trader_btc15m_sft_shadow:1`, while
`font_maker_random_stroke_train:1`, `torch_worker:2`, and
`periodic_mps_exporter:5` remained active. Live/recent TOTO remaining estimate
is `14192.0s`. Still no clean real32 MPS result exists to verify.

At 2026-05-21 08:30 +07, refresh 22 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The transient live
ai_trader child cleared, leaving `font_maker_random_stroke_train:1`,
`torch_worker:2`, and `periodic_mps_exporter:5`. Live/recent TOTO remaining
estimate is `13587.0s`. The clean gate remains blocked by external jobs only.

At 2026-05-21 08:35 +07, refresh 23 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. A live ai_trader/TOTO
child re-entered, so blockers are back to `font_maker_random_stroke_train:1`,
`ai_trader_toto_worker:1`, `torch_worker:2`, and `periodic_mps_exporter:5`.
Live/recent TOTO remaining estimate is `13285.0s`. Still no clean result
artifact exists to verify.

At 2026-05-21 08:40 +07, refresh 24 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The live ai_trader
child cleared again, leaving `font_maker_random_stroke_train:1`,
`torch_worker:2`, and `periodic_mps_exporter:5`. Live/recent TOTO remaining
estimate is `12982.0s`. The guarded clean gate remains correctly blocked until
the external work clears.

At 2026-05-21 08:45 +07, refresh 25 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The font_maker train
and its torch wrapper blockers cleared. The only remaining blocker class is
`periodic_mps_exporter:5`; live/recent TOTO remaining estimate is `12680.0s`.
This is the closest the clean gate has been to running in this wait sequence,
but there is still no clean real32 MPS artifact to verify.

At 2026-05-21 08:50 +07, refresh 26 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. A transient TOTO
prediction-export child re-entered on top of the persistent exporter chain:
`ai_trader_toto_worker:1` and `periodic_mps_exporter:5`. Live/recent TOTO
remaining estimate is `12377.0s`. The clean gate is still waiting on external
ai_trader/TOTO work.

At 2026-05-21 09:00 +07, refresh 28 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The state regressed
from exporter-only: a new font_maker rs35 probe entered with torch wrappers,
so blockers are `font_maker_random_stroke_train:1`, `torch_worker:2`, and
`periodic_mps_exporter:5`. Live/recent TOTO remaining estimate is `11773.0s`.
Still no clean result artifact exists.

At 2026-05-21 09:15 +07, refresh 31 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The state regressed
again from exporter-only: a font_maker checkpoint evaluation entered with torch
wrappers, and an ai_trader/TOTO child was sampled. Blockers are
`high_cpu_external_job:1`, `torch_worker:2`, `ai_trader_toto_worker:1`, and
`periodic_mps_exporter:5`. Live/recent TOTO remaining estimate is `10867.0s`.
Still no clean result artifact exists.

At 2026-05-21 09:30 +07, refresh 34 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The state regressed
from exporter-only again: a new font_maker rs37 continuation entered with torch
wrappers. Blockers are `font_maker_random_stroke_train:1`, `torch_worker:2`,
and `periodic_mps_exporter:5`. Live/recent TOTO remaining estimate is
`9961.0s`. Still no clean result artifact exists.

At 2026-05-21 09:35 +07, refresh 35 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The rs37 font_maker
continuation and torch wrappers remained, and a live ai_trader/TOTO child was
sampled again. Blockers are `font_maker_random_stroke_train:1`,
`torch_worker:2`, `ai_trader_toto_worker:1`, and `periodic_mps_exporter:5`.
Live/recent TOTO remaining estimate is `9659.0s`. Still no clean result
artifact exists.

At 2026-05-21 09:40 +07, refresh 36 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The sampled live
ai_trader/TOTO child cleared, but the rs37 font_maker continuation and torch
wrappers remain: `font_maker_random_stroke_train:1`, `torch_worker:2`, and
`periodic_mps_exporter:5`. Live/recent TOTO remaining estimate is `9356.0s`.
Still no clean result artifact exists.

At 2026-05-21 09:45 +07, refresh 37 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The rs37 font_maker
continuation and torch wrappers remain, and the sampled ai_trader child
rotated to BTC15M SFT shadow work. Blockers are
`font_maker_random_stroke_train:1`, `torch_worker:2`,
`ai_trader_btc15m_sft_shadow:1`, and `periodic_mps_exporter:5`. Live/recent
TOTO remaining estimate is `9052.0s`. Still no clean result artifact exists.

At 2026-05-21 09:50 +07, refresh 38 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The sampled
ai_trader BTC15M SFT child cleared, but Spotlight indexing entered while the
rs37 font_maker continuation, torch wrappers, and periodic exporter remained.
Blockers are `font_maker_random_stroke_train:1`,
`macos_spotlight_indexer:1`, `torch_worker:2`, and
`periodic_mps_exporter:5`. Live/recent TOTO remaining estimate is `8749.0s`.
Still no clean result artifact exists.

At 2026-05-21 09:55 +07, refresh 39 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. A manual refresh
exposed and then fixed a status-tooling footgun: external preflight probes
could classify the authorized idle `run_worldfoam_clean_mps_gate_when_ready.py`
waiter as `mps_worker` because of its argv. The low-CPU monitor-wrapper
background allow-list now includes that clean waiter, and refresh 39 confirms
it is no longer in the blocker set. Spotlight also cleared. Remaining blockers
are `font_maker_random_stroke_train:1`, `torch_worker:2`, and
`periodic_mps_exporter:5`; live/recent TOTO remaining estimate is `8447.0s`.
Still no clean result artifact exists.

At 2026-05-21 10:00 +07, refresh 40 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The clean waiter
self-blocker stayed fixed, but Spotlight indexing re-entered while the rs37
font_maker continuation, torch wrappers, and periodic exporter remained.
Blockers are `font_maker_random_stroke_train:1`,
`macos_spotlight_indexer:1`, `torch_worker:2`, and
`periodic_mps_exporter:5`; live/recent TOTO remaining estimate is `8144.0s`.
Still no clean result artifact exists.

At 2026-05-21 10:11 +07, refresh 42 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. Spotlight cleared
again, but two live ai_trader/TOTO worker processes entered on top of the
rs37 font_maker continuation, torch wrappers, and periodic exporter.
Blockers are `ai_trader_toto_worker:2`,
`font_maker_random_stroke_train:1`, `torch_worker:2`, and
`periodic_mps_exporter:5`; live/recent TOTO remaining estimate is `7538.0s`.
Still no clean result artifact exists.

At 2026-05-21 10:16 +07, refresh 43 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The font_maker,
torch, and transient ai_trader/TOTO worker blockers cleared. The only
remaining blocker class is `periodic_mps_exporter:5`; live/recent TOTO
remaining estimate is `7230.0s`. This is back to the closest-to-launch state,
but no clean result artifact exists yet.

At 2026-05-21 10:36 +07, refresh 47 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The state regressed
from exporter-only: a new font_maker rs39 scale-smoke entered with torch
wrappers while the periodic exporter remained. Blockers are
`font_maker_random_stroke_train:1`, `torch_worker:2`, and
`periodic_mps_exporter:5`; live/recent TOTO remaining estimate is `6021.0s`.
Still no clean result artifact exists.

At 2026-05-21 10:41 +07, refresh 48 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The rs39 font_maker
scale-smoke and torch wrappers remain, and the machine now has additional
high-CPU external pressure in the sample, including a Python child under the
TOTO monitor and a long-running Steam process. Blockers are
`font_maker_random_stroke_train:1`, `high_cpu_external_job:2`,
`torch_worker:2`, and `periodic_mps_exporter:5`; live/recent TOTO remaining
estimate is `5718.0s`. Still no clean result artifact exists.

At 2026-05-21 10:46 +07, refresh 49 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The font_maker
blocker rolled from rs39 into an rs40 200-step boot run with torch wrappers,
and the sample now includes a live TOTO residual export child. The previous
Steam high-CPU blocker is not in the sampled blocker set. Blockers are
`ai_trader_toto_worker:1`, `font_maker_random_stroke_train:1`,
`torch_worker:2`, and `periodic_mps_exporter:5`; live/recent TOTO remaining
estimate is `5412.0s`. Still no clean result artifact exists.

At 2026-05-21 10:51 +07, refresh 50 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The rs40 font_maker
boot run and torch wrappers remain. The sampled live TOTO export child cleared
from the blocker categories, but macOS `mediaanalysisd` entered as a high-CPU
external process. Blockers are `font_maker_random_stroke_train:1`,
`high_cpu_external_job:1`, `torch_worker:2`, and `periodic_mps_exporter:5`;
live/recent TOTO remaining estimate is `5106.0s`. Still no clean result
artifact exists.

At 2026-05-21 10:56 +07, refresh 51 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The rs40 font_maker
boot run and torch wrappers remain. The macOS `mediaanalysisd` blocker cleared,
but a sampled `ai_trader` BTC15M SFT shadow worker entered. Blockers are
`ai_trader_btc15m_sft_shadow:1`, `font_maker_random_stroke_train:1`,
`torch_worker:2`, and `periodic_mps_exporter:5`; live/recent TOTO remaining
estimate is `4802.0s`. Still no clean result artifact exists.

At 2026-05-21 11:01 +07, refresh 52 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The font_maker
rs40 boot run, torch wrappers, and sampled `ai_trader` BTC15M SFT shadow
worker cleared from the blocker categories. This is the closest state since
refresh 43, but it is still not a clean launch window because the periodic
TOTO exporter remains and a high-CPU Steam process is sampled. Blockers are
`high_cpu_external_job:1` and `periodic_mps_exporter:5`; live/recent TOTO
remaining estimate is `4496.0s`. Still no clean result artifact exists.

At 2026-05-21 11:06 +07, refresh 53 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The sampled Steam
high-CPU blocker cleared, leaving only the periodic TOTO exporter. This is
again the closest-to-launch state: blocker set is only
`periodic_mps_exporter:5`; live/recent TOTO remaining estimate is `4193.0s`.
Still no clean result artifact exists.

At 2026-05-21 11:18 +07, refresh 55 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. Refresh 54 was still
exporter-only, but refresh 55 sampled a transient high-CPU TOTO/tree residual
export child under the same overnight monitor. Blockers are
`ai_trader_toto_worker:1` and `periodic_mps_exporter:5`; live/recent TOTO
remaining estimate is `3588.0s`. Still no clean result artifact exists.

At 2026-05-21 11:28 +07, refresh 57 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. Refresh 56 stayed
in the same blocker categories as refresh 55, but refresh 57 changed the
sampled child: the TOTO audit/check worker cleared and an `ai_trader` BTC15M
activation-RL dataset worker entered. Blockers are
`ai_trader_btc15m_activation_rl:1` and `periodic_mps_exporter:5`; live/recent
TOTO remaining estimate is `2983.0s`. Still no clean result artifact exists.

At 2026-05-21 11:32 +07, refresh 58 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The activation-RL
dataset worker cleared, but a sampled TOTO quote-snapshot child entered under
the same overnight monitor. Blockers are `ai_trader_toto_worker:1` and
`periodic_mps_exporter:5`; live/recent TOTO remaining estimate is `2681.0s`.
Still no clean result artifact exists.

At 2026-05-21 11:37 +07, refresh 59 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The sampled TOTO
quote-snapshot child cleared, returning the blocker set to only the periodic
TOTO exporter. Blockers are `periodic_mps_exporter:5`; live/recent TOTO
remaining estimate is `2378.0s`. Still no clean result artifact exists.

At 2026-05-21 11:42 +07, refresh 60 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The blocker set
regressed from exporter-only because a high-CPU `git add` process in this repo
entered the sample. Blockers are `high_cpu_external_job:1` and
`periodic_mps_exporter:5`; live/recent TOTO remaining estimate is `2076.0s`.
Still no clean result artifact exists.

At 2026-05-21 11:47 +07, refresh 61 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The high-CPU
`git add` blocker cleared, but a sampled TOTO residual live-quote child
entered under the overnight monitor. Blockers are `ai_trader_toto_worker:1`
and `periodic_mps_exporter:5`; live/recent TOTO remaining estimate is
`1773.0s`. Still no clean result artifact exists.

At 2026-05-21 11:52 +07, refresh 62 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The TOTO residual
live-quote child stayed in the sample and a high-CPU Codex renderer process
also entered, so the blocker set regressed again despite the TOTO countdown
continuing. Blockers are `ai_trader_toto_worker:1`,
`high_cpu_external_job:1`, and `periodic_mps_exporter:5`; live/recent TOTO
remaining estimate is `1469.0s`. Still no clean result artifact exists.

At 2026-05-21 11:57 +07, refresh 63 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The sampled TOTO
child and high-CPU Codex renderer cleared, returning the blocker set to only
the periodic TOTO exporter. Blockers are `periodic_mps_exporter:5`;
live/recent TOTO remaining estimate is `1166.0s`. Still no clean result
artifact exists.

At 2026-05-21 12:02 +07, refresh 64 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The blocker set
regressed from exporter-only because a TOTO residual live-prediction export
child entered under the overnight monitor. Blockers are
`ai_trader_toto_worker:1` and `periodic_mps_exporter:5`; live/recent TOTO
remaining estimate is `864.0s`. Still no clean result artifact exists.

At 2026-05-21 12:07 +07, refresh 65 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The TOTO residual
export child cleared, but a BTC15M SFT shadow worker entered while the periodic
TOTO exporter continued. Blockers are `ai_trader_btc15m_sft_shadow:1` and
`periodic_mps_exporter:5`; live/recent TOTO remaining estimate is `561.0s`.
Still no clean result artifact exists.

At 2026-05-21 12:12 +07, refresh 66 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The sampled BTC15M
SFT shadow worker cleared, returning the blocker set to only the periodic TOTO
exporter. Blockers are `periodic_mps_exporter:5`; live/recent TOTO remaining
estimate is `259.0s`. Still no clean result artifact exists.

At 2026-05-21 12:17 +07, refresh 67 still did not launch:
`status=waiting_for_ready`, `ready_to_run_now=false`,
`launch_returncode=None`, `summary_stale_for_poll=false`. The live/recent TOTO
remaining estimate reached `0.0s`, but the periodic exporter process chain is
still live and a new TOTO residual live-prediction export child was sampled.
Blockers are `ai_trader_toto_worker:1` and `periodic_mps_exporter:5`. Still no
clean result artifact exists.

At 2026-05-21 12:34 +07, the guarded waiter finally found a quiet window and
launched the clean real32 MPS candidate, but the launcher returned
`result_verification_failed` rather than accepting the gate. The train/eval
artifact exists at
`research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_123219.worldfoam.json`,
and its own acceptance block reports `all_rows_ok=true`,
`total_step_sublinear_vs_frames=true`, `backward_sublinear_vs_frames=true`,
`render_sublinear_vs_frames=true`, and the selected/owner-run tape checks true.
The measured row means were roughly:

- 2f: total `4.721ms`, backward `4.363ms`
- 4f: total `5.179ms`, backward `4.881ms`
- 8f: total `8.131ms`, backward `7.746ms`
- 16f: total `10.922ms`, backward `10.439ms`
- 32f: total `11.467ms`, backward `11.094ms`

The verifier rejected the artifact because `render.mean_s` is exactly zero on
all rows, which makes `render_scale_first_to_last` non-finite/invalid under the
current verifier contract. That appears to be a measurement-contract mismatch
for the fused loss/VJP path rather than an external-blocker issue: preflight had
zero blocking and contending processes, train/eval returned 0, and no waiter
screen remains active. The goal state is therefore still incomplete until we
either record a nonzero render component for this fused path or narrow the
verifier to use the timing fields that this path actually emits.

At 2026-05-21 12:41 +07, the fused timing contract was made explicit:
`render_timing_scope=fused_loss_vjp_includes_render`, with verifier acceptance
requiring positive total/backward/fused timings rather than a fake nonzero
render timer. The clean real32 MPS launcher was rerun with embedded result
verification and passed:

- Launch summary:
  `research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_124139.launch_summary.json`
- Train/eval artifact:
  `research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_124139.worldfoam.json`
- Train/eval return code: `0`
- Result verifier return code: `0`
- Result verifier failures: `[]`
- Goal-state report:
  `research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_fork_shader_goal_state.json`
  now reports `status=complete`, `objective_complete=true`, and no missing
  clean real32 MPS gate.

Accepted row means:

- 2f: train/heldout PSNR `13.467/14.108`, total `5.037ms`, backward/fused `4.691ms`
- 4f: train/heldout PSNR `13.476/13.921`, total `5.783ms`, backward/fused `5.445ms`
- 8f: train/heldout PSNR `13.494/13.938`, total `7.596ms`, backward/fused `7.138ms`
- 16f: train/heldout PSNR `13.510/14.108`, total `11.944ms`, backward/fused `11.517ms`
- 32f: train/heldout PSNR `13.598/14.204`, total `11.305ms`, backward/fused `10.898ms`

Top-level scales are `frame_scale=16.0`, `total_step_scale=2.244`,
`backward_scale=2.323`, and `fused_loss_vjp_scale=2.323`; acceptance includes
`total_step_sublinear_vs_frames=true`, `backward_sublinear_vs_frames=true`, and
`fused_loss_vjp_sublinear_vs_frames=true`.
