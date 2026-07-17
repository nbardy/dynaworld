# Owner-Run Frame-Bitmask Shader Fork

## Context

The dense frame-select fork was correctness-green and directionally faster on
the contaminated site8 comparison, but it introduced a bad 16f storage shape:
`frame_change_index_i16` scales as `(track, frame>0)` and made 16f schema
storage worse than regular factorized (`74,046` vs `67,014` bytes in the first
comparison attempt).

The next fork was the compact selector idea from that result: keep direct
per-frame change selection, but replace the dense table with a per-track frame
bitmask and compute the sparse-change rank inside the shader.

## Implemented

- Added tape mode
  `owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid`.
- Added `_build_delta_frame_bitmask_i32(...)`, requiring `frame_count <= 31` and
  at most one sparse change per track/frame.
- Added Python op wrapper
  `endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only`.
- Added native Metal/C++/torch binding:
  `wf2_endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only_tensor`.
- The shader uses:
  - `track_frame_mask_i32[track]` for frame membership
  - `popcount(mask & ((1 << (frame + 1)) - 1)) - 1` for local sparse-change rank
  - `track_change_offsets_i16[track] + local_rank` for the selected change row
- Removed stale dense resident metadata for the bitmask path. The selected MPS
  tape keeps `track_change_offsets_i16` and `track_frame_mask_i32`, not
  `track_change_offsets_i32`, `frame_change_index_i16`, `change_frame_i16`, or
  `track_chunk_change_offsets_i16`.

## Tests

Commands run from the dynaworld root:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_framebitmask_removes_dense_frame_table research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_packed_recompute_nomid_matches_owner_run_fused_mse_on_moving_rays -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_owner_run_coeff_factorization research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed research_experiments.world_foam_lane2.test_probe_owner_run_boundary_tape -v
```

Results:

- Native rebuild passed.
- Targeted bitmask storage + moving-ray parity tests passed in `75.699s`.
- Focused owner-run suite passed 9 tests in `366.467s`.
- `git diff --check` passed on touched WorldFoam files.

## Smoke

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --frame-counts 16 --render-size 16 --site-count 8 --near 0.0 --far 3.5 --density 8.0 --invalid-epsilon 1.0e-7 --transmittance-threshold 1.0e-4 --steps 1 --warmup-steps 0 --optimizer-mode manual-vjp --tape-mode owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid --endpoint-record-source slow-owner-run --out-json /tmp/worldfoam_factorized_framebitmask_smoke.json
```

Artifact summary:

- `status=ok`
- `benchmark_environment.status=contended`
- schema storage `61,760` bytes
- topology storage `36,624` bytes
- non-coeff MPS resident storage `36,736` bytes
- resident keys include `track_change_offsets_i16` and `track_frame_mask_i32`
- resident keys no longer include stale `track_change_offsets_i32`

The timing row is not promotable because the machine was contended by a STAR UVT
run and other background work. Treat this as correctness/storage evidence only.

## Next

## Follow-up: comparison gate wiring

`compare_factorized_frameselect_gate.py` now has an `--include-framebitmask`
path. In that mode it runs regular factorized, dense frame-select, and
frame-bitmask with per-mode stable preflights, retrying the full attempt when a
candidate artifact ends contaminated. Unit coverage now checks:

- aggregate comparison chooses frame-bitmask when dense frame-select regresses
  storage
- dry-run emits the frame-bitmask command and artifact path
- frame-bitmask artifact contamination triggers a full-attempt retry

Fast validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py research_experiments/world_foam_lane2/test_compare_factorized_frameselect_gate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate -v
```

Result: 8 tests passed.

Live short-window attempt:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py --run-id 2026-05-19_factorized_selector_compare_blocked_live --include-framebitmask --stable-preflight-checks 1 --wait-for-benchmark-environment-ok --wait-timeout-s 1 --wait-interval-s 1 --max-comparison-attempts 3
```

It correctly stopped before training:

- `status=preflight_failed_before_regular`
- `candidate_labels=["frameselect", "framebitmask"]`
- `regular_preflight_failure_reason=benchmark_environment_never_clean`

Artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_blocked_live.factorized_frameselect_compare_summary.json
```

Interrupted clean-window attempt:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py --run-id 2026-05-19_factorized_selector_compare_clean_site8 --include-framebitmask --wait-for-benchmark-environment-ok --stable-preflight-checks 2 --wait-timeout-s 1800 --wait-interval-s 15 --max-comparison-attempts 3
```

This did not reach a usable three-way comparison. Attempts 1 and 2 ran only the
regular factorized row and then retried because the regular artifact ended with
`benchmark_environment.status=contended` after other MPS/Python work appeared.
Attempt 3 was still waiting for a clean preflight when we stopped to reflect.
The summary currently reports `status=waiting_for_preflight`,
`current_attempt_index=3`, and `comparison=null`.

Follow-up fix: `compare_factorized_frameselect_gate.py` now catches
Ctrl-C/SIGTERM around the attempt loop and rewrites the summary with
`status=interrupted`, `interrupted_at`, `interrupted_reason`, and the previous
status. This keeps future stopped clean-window waits from being mistaken for an
active benchmark process. Added unit coverage for a preflight interruption.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py research_experiments/world_foam_lane2/test_compare_factorized_frameselect_gate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate -v
```

Result: 9 tests passed.

The first bounded retry after that patch used run id
`2026-05-19_factorized_selector_compare_clean_site8_retry_interruptfix`. It
got two clean preflight snapshots and launched regular factorized, but unrelated
pytest and STAR UVT processes appeared before the end snapshot. The regular
artifact ended with `benchmark_environment.status=contended`, top-level
`status=failed`, `total_step_scale_first_to_last=8.953`, and
`backward_scale_first_to_last=9.913`; no frame-select or frame-bitmask rows
were launched.

That failed retry exposed another gate bug: a nonzero child exit with a written
contended JSON artifact was reported as `regular_train_failed` before the gate
loaded the artifact, so it did not classify the attempt as retryable
contamination. The gate now loads `out_json` on nonzero train exits when present,
records artifact status/environment, and retries only when the artifact is
benchmark-contaminated (`benchmark_environment.status != "background"`). Added
unit coverage for a nonzero regular artifact that is contaminated but retryable.

Validation after this fix:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py research_experiments/world_foam_lane2/test_compare_factorized_frameselect_gate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate -v
```

Result: 10 tests passed.

A second bounded retry used run id
`2026-05-19_factorized_selector_compare_clean_site8_retry2_nonzero_retryfix`.
Attempt 1 again reached regular factorized and had a good internal scaling
shape (`total_step_scale_first_to_last=1.317`,
`backward_scale_first_to_last=1.378`), but an unrelated BTC export appeared by
the end snapshot, so the regular artifact was correctly rejected as contended.
Attempt 2 then hit a narrower race: the parent preflight had two clean snapshots,
but an unrelated pytest job appeared before the child process's own
`--require-benchmark-environment-ok` start check. The child exited `2` before
writing `out_json`, and the compare gate treated that as a hard
`regular_train_failed`.

Follow-up fix: the compare gate now treats child exit `2` without an artifact as
retryable start-environment contamination. It also rewrites the summary after
non-retryable train failures so the current attempt records the mode status
before returning. Added unit coverage for this child-start contamination path.

Validation after this fix:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py research_experiments/world_foam_lane2/test_compare_factorized_frameselect_gate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate -v
```

Result: 11 tests passed.

The retry2 artifact also showed stale top-level summary fields:
`current_attempt_index=2` and `regular_train_status=2`, but
`regular_artifact_status=ok` / `regular_benchmark_environment_status=contended`
were still values from attempt 1. The gate now clears per-mode result fields
before each mode run, so a later child-start contamination cannot inherit stale
artifact metadata from an earlier attempt. Added a regression test for that
exact sequence.

Validation after this fix:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py research_experiments/world_foam_lane2/test_compare_factorized_frameselect_gate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate -v
```

Result: 12 tests passed.

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8.factorized_frameselect_compare_summary.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8.attempt1.regular_factorized.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8.attempt2.regular_factorized.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry_interruptfix.factorized_frameselect_compare_summary.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry_interruptfix.attempt1.regular_factorized.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry2_nonzero_retryfix.factorized_frameselect_compare_summary.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry2_nonzero_retryfix.attempt1.regular_factorized.json
```

Run a clean side-by-side ladder when the machine is quiet:

1. regular factorized
2. dense frame-select
3. frame-bitmask

Use stable preflights before each mode and reject/retry contaminated artifacts.
Only after a clean site8 result should the winner be repeated at site24/high-cap
and compared against STAR UVT.

## Retry3 reflection stop

I launched a third retry after the stale-field fix:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py --run-id 2026-05-19_factorized_selector_compare_clean_site8_retry3_stalefieldfix --include-framebitmask --wait-for-benchmark-environment-ok --stable-preflight-checks 2 --wait-timeout-s 1800 --wait-interval-s 15 --max-comparison-attempts 5
```

Attempt 1 produced regular factorized rows but ended contended. Attempt 2
produced a clean regular-factorized artifact:

```text
REGULAR2 env=background total_scale=1.744 backward_scale=1.263 storage_scale=1.748
F=2  total=2.303ms back=1.986ms psnr=13.374/15.269 store=17.73%
F=4  total=3.669ms back=2.263ms psnr=13.407/15.235 store=11.06%
F=8  total=2.894ms back=2.484ms psnr=13.336/15.278 store=6.72%
F=16 total=4.015ms back=2.509ms psnr=13.525/15.473 store=3.75%
```

The same attempt reached dense frame-select. It looked faster in absolute timing
but was not promotable because an unrelated pytest job contaminated the end
snapshot:

```text
FRAMESELECT2 env=contended total_scale=1.735 backward_scale=1.808 storage_scale=2.100 noncoeff_storage_scale=4.812
F=2  total=1.766ms back=1.491ms psnr=13.374/15.269 store=16.31%
F=4  total=2.755ms back=2.138ms psnr=13.407/15.235 store=10.47%
F=8  total=2.354ms back=1.968ms psnr=13.336/15.278 store=6.71%
F=16 total=3.064ms back=2.696ms psnr=13.525/15.473 store=4.15%
```

I stopped the gate while it was waiting for the next stable preflight so we could
reflect. The summary artifact remained `status=waiting_for_preflight` because
the signal hit the `rtk` wrapper, but the effective outcome is manual stop after
one clean regular artifact and one contaminated dense frame-select artifact. No
frame-bitmask side-by-side speed artifact was produced in retry3.

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry3_stalefieldfix.factorized_frameselect_compare_summary.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry3_stalefieldfix.attempt2.regular_factorized.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry3_stalefieldfix.attempt2.frameselect_factorized.json
```

## Retry11 frame-bitmask promotion

I added a narrow resume path to `compare_factorized_frameselect_gate.py`:
accepted regular/frame-select/frame-bitmask JSONs can now be reused if they are
`status=ok` and have a background benchmark environment, and
`--candidate-labels` can run only the missing candidate. The focused comparison
suite now covers accepted-artifact reuse, contaminated accepted-artifact
rejection, and framebitmask-only dry-run selection.

The frame-bitmask shader then got two hot-loop cleanups:

1. cache `track_frame_mask_i32[track_id]`,
   `track_change_offsets_i16[track_id]`, and
   `track_change_offsets_i16[track_id + 1]` once per track/chunk;
2. parallelize selector setup across the existing per-frame threads instead of
   having `local_frame == 0` serially fill every frame's `tg_source/begin/end`.

Validation:

```bash
rtk sh -lc 'cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace'
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py research_experiments/world_foam_lane2/test_compare_factorized_frameselect_gate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_framebitmask_removes_dense_frame_table research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_packed_recompute_nomid_matches_owner_run_fused_mse_on_moving_rays -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --frame-counts 2,4 --render-size 16 --site-count 8 --steps 1 --warmup-steps 0 --optimizer-mode manual-vjp --tape-mode owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid --endpoint-record-source slow-owner-run --out-json /tmp/worldfoam_framebitmask_parallel_setup_smoke.json
```

The first clean post-cache retry,
`2026-05-19_factorized_selector_compare_clean_site8_retry9_framebitmask_masksetup_quiet`,
was useful but not promotable: it fixed the old 4f total failure, but a clean 8f
row spiked to `1.131x` total and `1.171x` backward versus the accepted regular
artifact. That looked like timing instability or residual selector setup cost,
so I did the parallel setup edit instead of promoting it.

The final clean retry,
`2026-05-19_factorized_selector_compare_clean_site8_retry11_framebitmask_parallelsetup`,
passed:

```text
status=ok best_candidate=framebitmask env=background
max_total_ratio=0.884 max_backward_ratio=0.886
max_schema_ratio=0.973 max_topology_ratio=0.922 max_noncoeff_resident_ratio=0.923
F=2  total_ratio=0.809 backward_ratio=0.856 schema_ratio=0.973 total=2.124ms backward=1.865ms
F=4  total_ratio=0.884 backward_ratio=0.886 schema_ratio=0.947 total=2.133ms backward=1.870ms
F=8  total_ratio=0.809 backward_ratio=0.812 schema_ratio=0.930 total=2.008ms backward=1.763ms
F=16 total_ratio=0.878 backward_ratio=0.876 schema_ratio=0.922 total=2.182ms backward=1.918ms
```

This promotes frame-bitmask as the current site8 WorldFoam selector winner over
regular factorized. It does not yet prove site24/high-cap competitiveness or
STAR UVT parity.

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry9_framebitmask_masksetup_quiet.factorized_frameselect_compare_summary.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry11_framebitmask_parallelsetup.factorized_frameselect_compare_summary.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry11_framebitmask_parallelsetup.attempt1.framebitmask_factorized.json
```

## Site24 high-cap repeat

I ran the matched site24/high-cap WorldFoam gate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py --run-id 2026-05-19_factorized_selector_compare_clean_site24_retry1_framebitmask_parallelsetup --candidate-labels framebitmask --site-count 24 --render-size 16 --frame-counts 2,4,8,16 --steps 6 --warmup-steps 2 --wait-for-benchmark-environment-ok --stable-preflight-checks 2 --wait-timeout-s 1800 --wait-interval-s 15 --max-comparison-attempts 6
```

Regular factorized attempt 1 was clean. Frame-bitmask attempt 1 had good rows
but was rejected because an unrelated `ai_trader` export spiked during the end
snapshot. Frame-bitmask attempt 2 was clean and the summary passed:

```text
status=ok best_candidate=framebitmask env=background
max_total_ratio=0.942 max_backward_ratio=0.941
max_schema_ratio=0.978 max_topology_ratio=0.940 max_noncoeff_resident_ratio=0.940
F=2  total_ratio=0.854 backward_ratio=0.869 schema_ratio=0.978 total=2.007ms backward=1.761ms
F=4  total_ratio=0.916 backward_ratio=0.941 schema_ratio=0.952 total=2.191ms backward=1.929ms
F=8  total_ratio=0.864 backward_ratio=0.856 schema_ratio=0.927 total=2.064ms backward=1.806ms
F=16 total_ratio=0.942 backward_ratio=0.927 schema_ratio=0.909 total=2.195ms backward=1.922ms
```

This extends the frame-bitmask promotion from site8 to site24/high-cap for the
synthetic WorldFoam selector comparison. The remaining competitiveness gate is a
matched STAR UVT speed comparison.

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site24_retry1_framebitmask_parallelsetup.factorized_frameselect_compare_summary.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site24_retry1_framebitmask_parallelsetup.attempt1.regular_factorized.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site24_retry1_framebitmask_parallelsetup.attempt2.framebitmask_factorized.json
```
