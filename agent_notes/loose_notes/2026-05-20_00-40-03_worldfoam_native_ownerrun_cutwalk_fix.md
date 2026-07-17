# WorldFoam native owner-run cutwalk fix

Context: selected-only framebitmask prep removed baseline segment-tape and
compaction accounting, but the remaining Python `slow-owner-run` endpoint
sequence construction still dominated wall time (`54.09s` at 4f in the
selected-only smoke, and `221.47s` at 16f in the previous full-metric artifact).

What failed first:

- The first native cutwalk prototype reused
  `gate4_delta_replace_from_cuts_cpu`.
- That op assumes every crossed boundary involving the current owner is an
  active transition.
- That is not true for full power-diagram cut lists: inactive cell boundaries
  can be crossed before the nearest-site owner changes.
- The new parity test caught this as a shader loss mismatch of about
  `7.75e-4`, well above the `5e-6` tolerance.

Fix:

- Added `gate4_owner_run_delta_replace_from_rays_cpu` in the
  `world_foam_lane2_fused_slab_v0` C++ binding.
- The op takes boundary coefficients, site `(x,y,z,t,weight)`, site densities,
  rays, and frame indices.
- It computes boundary cut depths in C++, dedupes them with the same internal
  depth rule as Python, computes midpoint nearest-site owner per interval,
  merges consecutive same-owner intervals, and applies the same transmittance
  threshold before emitting endpoint records.
- `--experimental-native-owner-run-cutwalk-delta` now calls this exact op
  instead of building Python cut arrays.

Validation:

```bash
rtk sh -lc 'cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace'
```

Build passed.

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.NativeOwnerRunCutwalkCpuTests.test_native_cutwalk_delta_matches_python_owner_run_sequences \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_native_cutwalk_framebitmask_matches_python_sequence_shader_output \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_selected_only_framebitmask_prep_skips_baseline_segment_tape \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_framebitmask_removes_dense_frame_table \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_packed_recompute_nomid_matches_owner_run_fused_mse_on_moving_rays -v
```

Result: the full owner-run delta packed suite passed `8` tests in `561.792s`.

Path smokes:

- `2026-05-20_worldfoam_framebitmask_render64_site24_selected_only_native_cutwalk_4f_smoke.json`
  is `status=ok`; environment was contended, but the path is correct. Train
  endpoint sequence prep is `0.553s`, versus `54.09s` for the prior Python
  selected-only 4f smoke.
- `2026-05-20_worldfoam_framebitmask_render64_site24_selected_only_native_cutwalk_2_4_8_16_path.json`
  is `status=ok` with environment status `background` and all acceptance keys
  true. Train sequence prep is `0.190/0.693/1.261/1.329s` for `2/4/8/16f`.
  Backward medians are `2.304/2.749/4.845/3.739ms`; total medians are
  `2.630/3.130/5.510/4.145ms`.
- A follow-up `steps=8/warm4` run with `--require-benchmark-environment-ok`
  correctly refused to start because the preflight saw
  `scripts.probe_btc15m_tree_oracle_context_feature_frame` at about `95%` CPU.
  A later preflight was still contended by a font-experiment pytest and another
  Python process, so no clean promotion artifact was written.
- A later `clean_retry2` run wrote
  `2026-05-20_worldfoam_framebitmask_render64_site24_selected_only_native_cutwalk_2_4_8_16_clean_retry2.json`
  with completed rows but `benchmark_environment.status=contended`. Treat it as
  diagnostic only. Its backward medians were `2.278/2.738/2.824/3.844ms`, total
  medians were `2.568/3.170/3.141/4.230ms`, train prep was
  `0.117/0.342/0.683/1.846s`, and PSNR was `11.770/11.783/12.150/12.248`.
- `clean_retry3` started from a promotable preflight (`background`, no blocking
  processes) but exited `2` because the end snapshot became contended. The
  blockers were an unrelated `ai_trader` export wrapper at about `13.8%` CPU
  plus a low-CPU font-training `torch` wrapper. Treat the artifact as
  diagnostic only. Its backward medians were `2.698/22.295/3.980/6.014ms`,
  total medians were `3.137/28.368/4.790/6.541ms`, train prep was
  `0.248/0.823/1.227/2.803s`, and PSNR was `11.770/11.783/12.150/12.248`.
- Tightened `compare_star_uvt_worldfoam_scale.py` with
  `--require-clean-worldfoam-artifact` and
  `--require-benchmark-environment-ok`, and added a focused unit test proving a
  contended WorldFoam artifact is rejected before making a STAR competitiveness
  claim.
- Added `--wait-for-benchmark-environment-ok-timeout-s` /
  `--wait-for-benchmark-environment-ok-poll-s` to both the WorldFoam ladder and
  STAR comparison runners. A focused STAR comparison unit test proves a
  contended-then-clean preflight waits and then proceeds. A live 30s WorldFoam
  wait smoke waited through the timeout and exited `2` because unrelated
  dynamic-gsplat, `ai_trader`, font-training, and git-add work stayed
  contending.
- Added `run_worldfoam_star_native_cutwalk_gate.py`, an unattended wrapper for
  the remaining clean promotion gate. It runs the guarded native-cutwalk
  render64/site24 WorldFoam ladder first, rejects failed/contended artifacts,
  then runs the guarded matched STAR comparison at 64px/896 tubes. Unit
  coverage proves dry-run command construction and labels a child exit `2`
  without an artifact as `worldfoam_preflight_failed_or_contended`. A short
  live blocked smoke wrote
  `2026-05-20_native_cutwalk_worldfoam_star_blocked_smoke.promotion_summary.json`
  and did not start STAR because the WorldFoam preflight stayed contended.
- Added `test_train_eval_owner_run_benchmark_environment.py` so the WorldFoam
  wait helper itself has focused coverage. The focused wait/wrapper/STAR
  comparison unit set now passes `13` tests, covering WorldFoam wait success,
  zero-timeout contended return, wrapper dry-run construction, preflight-blocked
  labeling, contended-artifact labeling, STAR wait behavior, and contended
  WorldFoam artifact rejection.
- Bounded wrapper attempt
  `2026-05-20_native_cutwalk_worldfoam_star_wait_attempt1` waited through an
  initially contended machine, caught a clean start preflight, and completed all
  WorldFoam rows. The end snapshot became `contended` after unrelated
  font-training and `ai_trader` jobs appeared, so the wrapper did not run STAR.
  Treat the WorldFoam artifact as diagnostic only. Its backward medians were
  `2.700/2.738/3.044/5.892ms`, total medians were
  `3.116/3.240/3.651/6.865ms`, train prep was
  `0.295/0.592/1.199/2.452s`, and train PSNR was
  `11.770/11.783/12.150/12.248`.
- Added `--max-worldfoam-attempts` to the wrapper. It retries WorldFoam when a
  preflight times out or an artifact completes with end-snapshot contention,
  and the STAR comparison is rebuilt to consume the first clean WorldFoam
  artifact path. A unit test covers a contended attempt followed by a clean
  attempt and then STAR.
- Continuation verification re-ran the focused wrapper/wait/STAR comparison
  unit set:

  ```bash
  rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
    research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate \
    research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment \
    research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale -v
  ```

  Result: `13` tests passed in `0.023s`. `py_compile` and `git diff --check`
  also passed for the touched runner/test/doc files.
- The same continuation checked the live benchmark environment before launching
  a timing gate. It was still `contended`, with an unrelated `font_maker` Torch
  process around `126%` CPU and a git-add process around `38%` CPU. No
  WorldFoam/STAR timing run was launched because it would not be promotable.
  A follow-up preflight after the git-add load cleared was still `contended`
  because the same `font_maker` Torch process was around `108%` CPU.
- Hardened the wrapper so it runs `--benchmark-environment-check-only` before
  each WorldFoam attempt and stores the full preflight environment snapshot in
  the promotion summary. The focused wrapper/wait/STAR comparison suite still
  passes `13` tests after this change. A live zero-timeout audit,
  `2026-05-20_native_cutwalk_worldfoam_star_preflight_audit`, exited
  `worldfoam_preflight_failed_or_contended` without launching WorldFoam and
  wrote the blocker snapshot: an `ai_trader` feature-context build around
  `122%` CPU plus a pytest process around `96%` CPU.
- Added `--preflight-only` to the wrapper for intentional readiness checks that
  should never launch WorldFoam or STAR. The focused wrapper/wait/STAR
  comparison suite now passes `14` tests. Live preflight-only audit
  `2026-05-20_native_cutwalk_worldfoam_star_preflight_only_audit` exited
  `worldfoam_preflight_failed_or_contended`, kept `worldfoam_attempts=[]`, and
  wrote the current blocker snapshot: `font_maker` training around `100%` CPU,
  `ai_trader` SFT/pytest around `74%` CPU, and a STAR UVT feature-overfit run
  around `62%` CPU.
- Added compact `worldfoam_preflight_blocking_processes` summaries at the
  promotion-summary top level and per attempt, so a blocked handoff can be read
  without digging through the full process snapshot. The focused
  wrapper/wait/STAR comparison suite still passes `14` tests after this change.
  Live compact recheck
  `2026-05-20_native_cutwalk_worldfoam_star_preflight_compact_recheck` kept
  `worldfoam_attempts=[]` and recorded `font_maker` training around `130%` CPU
  plus pytest around `95%` CPU as the current blockers.
- Refined the compact blocker summary to keep only actual high-CPU blockers when
  any are present; the full snapshot still keeps all blocking rows, including
  hard-keyword parent wrappers. The focused wrapper/wait/STAR comparison suite
  now passes `15` tests. Live filtered recheck
  `2026-05-20_native_cutwalk_worldfoam_star_preflight_filtered_continue` kept
  `worldfoam_attempts=[]` and recorded compact blockers as `font_maker`
  training around `113%` CPU and an `ai_trader` Toto residual export child
  around `49%` CPU.
- Resume preflight-only audit
  `2026-05-20_native_cutwalk_worldfoam_star_preflight_resume` still exited
  `worldfoam_preflight_failed_or_contended` without launching WorldFoam. The hot
  blockers were unrelated `font_maker` training around `140%` CPU, a STAR
  dense-alpha diagnostic around `88.5%`, `ai_trader` Toto live-quote shadow
  paper around `80%`, and git add around `37.6%`.
- Hardened the wrapper summary contract after the latest failed multi-attempt
  run exposed a misleading handoff field: top-level `worldfoam_artifact` now
  means the clean promotable WorldFoam artifact selected for STAR. Failed
  preflights leave it `null`; diagnostic paths are explicit as
  `planned_worldfoam_artifact`, `worldfoam_latest_attempt_artifact`, and
  `worldfoam_latest_written_artifact`. Per-attempt rows also record
  `artifact_written` and `promotable`. `py_compile`, `git diff --check`, and the
  focused wrapper/wait/STAR comparison suite passed after the change:
  `15` tests in `0.030s`.
- Post-change live audit
  `2026-05-20_native_cutwalk_worldfoam_star_preflight_contract_resume` proved the
  new preflight-only summary shape under real contention:
  `worldfoam_artifact=null`, `worldfoam_attempts=[]`, no WorldFoam launch. The
  compact blockers were `font_maker` training around `137%` CPU and `ai_trader`
  SFT shadow around `18.5%` CPU.
- A later quick preflight-only check
  `2026-05-20_native_cutwalk_worldfoam_star_preflight_now` was also contended
  and did not launch WorldFoam. Compact blockers were `font_maker` training
  around `120.7%` CPU, an `ai_trader` tree-residual export child around `86.4%`,
  and git add around `35.4%`.
- Added the missing regression for the exact stale-artifact path seen in the
  live multi-attempt wrapper: attempt 1 writes a contended diagnostic
  WorldFoam artifact, attempt 2 fails preflight before writing anything, and the
  summary must keep top-level `worldfoam_artifact=null`, retain
  `worldfoam_latest_written_artifact` at attempt 1, and move
  `worldfoam_latest_attempt_artifact` to attempt 2. `py_compile` passed and the
  focused wrapper/wait/STAR comparison suite now passes `16` tests in `0.038s`.
- Split STAR compare command fields to avoid another misleading handoff:
  `planned_star_compare_command` records the audit-plan command, while selected
  `star_compare_command` stays `null` until a clean WorldFoam artifact is
  actually passed to STAR. Wrapper-local tests pass after the change. Live
  preflight-only audit
  `2026-05-20_native_cutwalk_worldfoam_star_preflight_starcmd_contract` proved
  the new shape under real contention: `worldfoam_artifact=null`,
  `star_compare_command=null`, `worldfoam_attempts=[]`, no WorldFoam launch.
  Compact blockers were `ai_trader` SFT shadow around `81.9%` CPU and a STAR
  dense-alpha diagnostic around `45.4%` CPU.
- Added `summary_schema_version=worldfoam_star_native_cutwalk_gate_v2` to the
  promotion summary so old pre-contract artifacts cannot be mistaken for the
  tightened contract. `py_compile` passed, and the focused wrapper/wait/STAR
  comparison suite still passes `16` tests. Live preflight-only audit
  `2026-05-20_native_cutwalk_worldfoam_star_preflight_schema_contract` wrote the
  v2 summary, exited `worldfoam_preflight_failed_or_contended`, kept
  `worldfoam_artifact=null`, `star_compare_command=null`, and
  `worldfoam_attempts=[]`, and did not launch WorldFoam or STAR. The compact
  blocker list showed a hot `font_maker` torch child around `152.6%` CPU; the
  `ai_trader` shadow monitor was visible but idle in the background snapshot.
- Bounded full wrapper attempt
  `2026-05-20_native_cutwalk_worldfoam_star_clean_wait120` waited through three
  120s preflight attempts and still exited
  `worldfoam_preflight_failed_or_contended`. No WorldFoam artifact was written,
  `worldfoam_latest_written_artifact=null`, `worldfoam_artifact=null`, and
  `star_compare_command=null`, so no contaminated speed rows were produced.
  Attempt blockers stayed external: `font_maker` torch remained hot around
  `139-156%` CPU and intermittent `ai_trader` SFT children hit about `77-95%`
  CPU.
- Added an all-preflight-failures regression matching that live blocked shape:
  three contended preflights must not call WorldFoam, must keep
  `worldfoam_latest_written_artifact=null`, must leave
  `worldfoam_artifact=null`, and must keep `star_compare_command=null` while
  preserving the final compact blocker. `py_compile` passed and the focused
  wrapper/wait/STAR comparison suite now passes `17` tests in `0.027s`.
- Added CPU parity coverage for the moving first-person/multiview contract:
  the native owner-run cutwalk delta builder now matches the Python owner-run
  sequence builder when the same frame sequence is duplicated with a shifted ray
  block. This protects the sample-order/view-major path rather than only the
  old single-view moving-ray fixture. The focused native cutwalk CPU class
  passes `2` tests in `28.622s`.
- Added the matching MPS shader-output regression for that duplicated shifted
  multiview fixture. The source fixture already contains two views, so the
  regression asserts the duplicated view count is `2 * base_view_count` instead
  of hard-coding two. The selected framebitmask native-cutwalk shader output and
  site gradients match the Python sequence-built tape on both the original and
  duplicated multiview moving-ray fixtures (`2` focused MPS tests, `27.706s`).
- Live wrapper preflight-only audit
  `2026-05-20_native_cutwalk_worldfoam_star_preflight_font_blocked` still exited
  `worldfoam_preflight_failed_or_contended` without launching WorldFoam or STAR.
  It wrote schema v2, kept `worldfoam_artifact=null` and
  `star_compare_command=null`, and recorded the compact blocker as unrelated
  font-flow training around `163.4%` CPU. The focused wrapper/wait/STAR suite
  still passes `17` tests in `0.023s`.
- Added `verify_worldfoam_star_native_cutwalk_promotion.py` as the final
  acceptance audit for the clean gate. It rejects stale/preflight-only summaries,
  missing selected artifacts, contended WorldFoam artifacts, contended STAR
  comparison artifacts, wrong schema, missing native-cutwalk flags, and
  mismatched STAR-consumed WorldFoam paths. Its focused unit tests pass (`3`
  tests in `0.007s`), and running it on the current blocked preflight summary
  correctly exits `2`.
- Wired that audit into `run_worldfoam_star_native_cutwalk_gate.py` behind
  `--verify-promotion`. On a successful STAR comparison the wrapper writes the
  summary, runs the verifier, records `promotion_verifier_command`,
  `promotion_verifier_returncode`, `promotion_verifier_status`, and
  `promotion_verifier_failures`, and changes the final status to
  `promotion_verification_failed` if the audit fails. Wrapper-local plus
  verifier unit coverage passes `13` tests after this change.
- Live preflight-only audit
  `2026-05-20_native_cutwalk_worldfoam_star_preflight_verifier_integration_blocked`
  exercised the updated wrapper command shape and correctly exited
  `worldfoam_preflight_failed_or_contended` without launching WorldFoam or STAR.
  The summary is schema v2, keeps `worldfoam_artifact=null` and
  `star_compare_command=null`, and records compact blockers as unrelated
  `font_maker` training (`~155.2%` CPU) plus an `ai_trader` activation RL
  dataset build (`~99.4%` CPU).
- Continuation validation rechecked the live benchmark environment and it
  remains contended, now by `font_maker` torch (`~157.3%` CPU) plus an
  `ai_trader` pytest training child (`~98.7%` CPU), so no clean timing gate was
  launched. The non-timing contracts are green: the wrapper/wait/STAR/verifier
  suite passes `22` tests, the native cutwalk CPU parity class passes `2` tests
  in `25.409s`, and selected framebitmask native-cutwalk MPS shader-output
  parity passes `2` tests in `27.771s`.
- Added opt-in stable preflight sampling to the promotion wrapper:
  `--preflight-stability-samples` plus
  `--preflight-stability-interval-s`. This requires consecutive clean
  benchmark-environment samples before launching WorldFoam, records
  `worldfoam_preflight_samples`, and makes the next clean gate less likely to
  start during a short quiet gap before another local job appears. Added a
  late-contention regression where sample 1 is clean and sample 2 is contended;
  it must not launch WorldFoam. The broader wrapper/wait/STAR/verifier suite
  now passes `23` tests. Dry run
  `/tmp/worldfoam_stable_preflight_dryrun.json` shows the strict command shape
  with `3` samples and `5s` spacing. Live
  `2026-05-20_native_cutwalk_stable_preflight_blocked` requested that guard and
  correctly exited `worldfoam_preflight_failed_or_contended` on sample 1, with
  `worldfoam_artifact=null`, `star_compare_command=null`, and compact blockers
  from `font_maker` torch (`~165.8%` CPU) plus an `ai_trader` tree-oracle
  context probe (`~7.5%` CPU).

Interpretation:

- The exact native builder fixes the semantic bug and removes the dominant
  Python endpoint-record construction cost.
- The 2/4/8/16 path ladder has now cleared the quiet-window wrapper and
  integrated verifier. `2026-05-20_native_cutwalk_worldfoam_star_starretry`
  exited `status=ok` with one clean/promotable WorldFoam attempt, one
  clean/promotable STAR attempt, both benchmark environments `background`, and
  `promotion_verifier_status=ok`.

## Strict-background STAR retry gate

Follow-up wrapper/verifier hardening:

- Promotion now requires `benchmark_environment.status=background` exactly.
  Missing environments and older ambiguous `ok` environments are not clean.
- The wrapper now has `--max-star-attempts` and retries only a contended STAR
  comparison without rerunning a clean WorldFoam artifact.
- STAR retry artifacts are named
  `<run_id>.star_attemptN.star_compare.json`; the summary records
  `star_compare_attempts`, `max_star_attempts`,
  `star_compare_latest_attempt_artifact`, and
  `star_compare_latest_written_artifact`.
- Added regressions for missing WorldFoam environment, missing STAR environment,
  and STAR-only contention retry. Added verifier coverage that `ok` is not a
  clean benchmark-environment status.

Validation:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py \
  research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py \
  research_experiments/world_foam_lane2/test_run_worldfoam_star_native_cutwalk_gate.py \
  research_experiments/world_foam_lane2/test_verify_worldfoam_star_native_cutwalk_promotion.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion \
  research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment \
  research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale -v
```

Result: `27` broader wrapper/verifier/benchmark-environment/STAR comparison
tests pass.

Successful live gate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py \
  --run-id 2026-05-20_native_cutwalk_worldfoam_star_starretry \
  --wait-timeout-s 300 \
  --wait-poll-s 30 \
  --max-worldfoam-attempts 3 \
  --max-star-attempts 3 \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 \
  --verify-promotion
```

Artifacts:

- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_starretry.promotion_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_starretry.attempt1.worldfoam.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_starretry.star_attempt1.star_compare.json`

Result details:

- WorldFoam artifact: `status=ok`, `benchmark_environment.status=background`,
  all acceptance flags true.
- WorldFoam artifact means for `2/4/8/16f`: total
  `3.008/3.014/3.323/4.095ms`; backward `2.739/2.517/2.561/3.796ms`;
  total/backward scale `1.361x/1.386x`; train PSNR
  `11.770/11.783/12.150/12.248`; heldout PSNR
  `12.352/12.406/12.589/12.857`.
- STAR comparison artifact: `status=ok`,
  `benchmark_environment.status=background`, no failures.
- STAR medians for `2/4/8/16f`: total `5.003/5.943/8.092/9.794ms`;
  backward `2.629/3.411/5.083/6.768ms`; total/backward scale
  `1.957x/2.574x`.
- Comparison medians: STAR/WorldFoam total ratios are
  `1.95/2.25/2.77/2.61`; backward ratios are `1.14/1.45/2.00/1.96`.

Interpretation update:

- The clean Gate4 native-cutwalk fused-MSE micro-gate is now faster than the
  matched STAR micro-gate on both total step and backward at all four tested
  frame counts. The dated micro-gate rows were added to `BASELINES.md`.
- This is not full STAR UVT system parity. It is a render64/site24 fused-MSE
  speed/scale row. STAR still owns the stronger RGB-quality lineage and broader
  training route.
- The next useful decision is whether to record this micro-gate in
  `BASELINES.md` as a synthetic speed baseline or demand a stronger fixture
  first: larger render/site pressure, longer frame ladder, or a bridge that
  connects fused-MSE Gate4 speed to a real RGB/feature quality objective.

## Repeat-32 synthetic fixture gate

The first 32f extension without repeat mode failed for the right reason: the
real fixture only loads 16 frames, and the STAR probe now refuses to silently
pretend that 32 distinct frames were measured. `--repeat-loaded-frames` is
therefore explicit and labeled as a synthetic repeated-fixture speed-scaling
smoke.

The first repeated 32f WorldFoam attempt then exposed a real framebitmask
boundary: the host/shader wrapper rejected `frame_count=32` because the
selected-track frame mask used signed int32 bits and the old guard capped at
`<=31`. The fix now allows exactly 32 frames by storing bit 31 as the signed
int32 payload and validating masks as unsigned bit patterns; `frame_count > 32`
still rejects. No Metal kernel math change was needed because the kernel casts
`track_frame_mask_i32` to `uint` before testing `1u << global_frame_id`.

Validation for the framebitmask fix:

```bash
rtk sh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.NativeOwnerRunCutwalkCpuTests \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_framebitmask_removes_dense_frame_table -v
```

Result: rebuild passed; the focused post-build runtime tests passed (`5` tests
in `82.712s`).

Successful repeated-fixture gate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py \
  --run-id 2026-05-20_native_cutwalk_worldfoam_star_repeat32_framebitmask32fix \
  --frame-counts 2,4,8,16,32 \
  --repeat-loaded-frames \
  --wait-timeout-s 900 \
  --wait-poll-s 15 \
  --max-worldfoam-attempts 3 \
  --max-star-attempts 3 \
  --preflight-stability-samples 3 \
  --preflight-stability-interval-s 5 \
  --verify-promotion
```

Artifacts:

- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_repeat32_framebitmask32fix.promotion_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_repeat32_framebitmask32fix.attempt2.worldfoam.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_repeat32_framebitmask32fix.star_attempt2.star_compare.json`

Result details:

- Promotion summary: `status=ok`, WorldFoam and STAR benchmark environments
  both `background`, integrated verifier `status=ok`, and
  `repeat_loaded_frames=true`.
- WorldFoam 32f row explicitly records `loaded_frame_count=16`,
  `repeat_loaded_frames=true`, and
  `repeat_loaded_frames_scope=synthetic repeated-fixture speed-scaling smoke`.
- STAR 32f row explicitly records `requested_frames=32`,
  `loaded_frame_count=16`, and `repeat_loaded_frames_used=true`.
- WorldFoam medians for `2/4/8/16/32f`: total
  `2.829/3.248/4.414/4.643/6.371ms`; backward
  `2.557/2.965/4.054/4.254/6.001ms`; total/backward scale
  `2.252x/2.347x` over requested `2 -> 32f`.
- STAR medians for the same requested frame ladder: total
  `5.324/6.436/7.623/9.937/13.344ms`; backward
  `2.770/3.495/4.474/6.126/9.013ms`; total/backward scale
  `2.506x/3.254x`.
- The comparison still has WorldFoam faster at every requested frame count in
  this micro-gate. At 32f, STAR/WorldFoam is `2.095x` on total median and
  `1.502x` on backward median.
- WorldFoam mixed selected-tape storage grows `7.562x` over the requested
  `16x` frame-count increase, so it is still sublinear storage-wise, but this
  is not constant-memory and the 32f point is synthetic because it repeats a
  16-frame loaded target.

Follow-up verification:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py \
  research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_repeat32_framebitmask32fix.promotion_summary.json

rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_worldfoam_star_native_cutwalk_promotion \
  research_experiments.world_foam_lane2.test_run_worldfoam_star_native_cutwalk_gate \
  research_experiments.world_foam_lane2.test_compare_star_uvt_worldfoam_scale \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_benchmark_environment \
  research_experiments.world_foam_lane2.test_star_uvt_timing_probe_frame_fit \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.NativeOwnerRunCutwalkCpuTests \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_framebitmask_removes_dense_frame_table -v
```

Result: verifier passed with no failures; the focused suite passed `35` tests
in `51.172s`.

Interpretation:

- The 32-bit framebitmask path is fixed and covered for exactly 32 requested
  frames.
- The comparison harness now prevents silent 16-frame-vs-32-frame confusion for
  STAR and future WorldFoam compare summaries. In the current repeat32 artifact,
  the authoritative WorldFoam repeat metadata lives in the WorldFoam rows.
- This strengthens the speed-scaling story, but it is still not a real 32-frame
  quality/system baseline. The next stronger evidence should use a real
  longer-than-16f fixture or a larger render/site/quality-linked gate rather
  than another repeated-fixture point.

Continuation check on 2026-05-20:

- Re-ran the repeat32 promotion verifier after the wording/summary-metadata
  cleanup; it returned `status=ok` with no failures.
- Re-ran the focused comparison/unit gate above; it passed `35` tests in
  `49.453s`.
- Re-ran the actual native-cutwalk framebitmask MPS shader-output subset
  (`test_native_cutwalk_framebitmask_matches_python_sequence_shader_output`,
  `test_native_cutwalk_framebitmask_shader_output_matches_python_for_multiview_moving_rays`,
  and `test_selected_only_framebitmask_prep_skips_baseline_segment_tape`); it
  passed `3` tests in `21.500s`.
- `py_compile` passed for `compare_star_uvt_worldfoam_scale.py` and
  `test_compare_star_uvt_worldfoam_scale.py`.
- `git diff --check` passed for the touched WorldFoam/STAR files and synced
  docs.
- `agent_notes/key_learnings.md` is already at its 199-line budget; do not add a
  new bullet unless recompressing older WorldFoam/STAR entries first.
