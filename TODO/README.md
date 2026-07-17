# DynaWorld TODO Index

This folder is the active backlog. Use this index to route work before opening
individual TODO files.

## Current Project State

2026-07-18 STAR UVT implementation/benchmark closeout: thin trainer wrappers
now share `train_cli.run_config_main(...)`, with focused CLI/registry tests
passing. Fresh RGB STAR kernel evidence is recorded in
`../agent_notes/loose_notes/2026-07-18_00-02-37_star_uvt_kernel_refresh_and_cli_completion.md`
and appended to `../BASELINES.md`. The default remains `direct_atomic +
index_add`; 512px direct-serial is a promising kernel-probe result only and
needs a trainer-level parity/repeat gate before promotion. The renderer-scaling
report now tolerates archived-source absence and ingests the fresh STAR matrix,
but a true current dynamic-gsplat/F32 comparison needs new matching raster
benchmarks because the old source JSONLs are absent. Do not repeat the
alpha-background ablation: its current renderer/resolution-specific decision
is already recorded in `EXPERIMENTS.md` and the May 21 loose notes.

2026-07-17 closeout decision: stop adding umbrella Gauged UVT/fiber-bundle
theory. Keep World Tubes as the primary compiled camera-program implementation
and paper lane. Keep WorldFoam as a parked retained-depth optical-transfer
challenger, not a parallel default. The next admissible World Tubes work is
camera-triplet/scene breadth, orbit/visibility stress that falsifies a current
certificate, or measured native-kernel overhead. The next admissible WorldFoam
work requires broader heldout quality or native optical-transfer parity.

As of 2026-05-28:

- First operational map for new agents lives in `../PROJECT_INDEX.md`.
- Evergreen thread types live in `../.agents/thread_types/`, including the
  hourly CTO code reviewer.
- Active experiment registry lives in `../EXPERIMENTS.md`.
- Code organization and deduplication roadmap lives in
  `../CODE_ORGANIZATION.md`.
- The public progress checklist lives in `../README.md`.
- Current measured standings and missing benchmark rows live in
  `../BASELINES.md`.
- The data-loader contract lives in `../research_notes/data_contract.md`.
- The dense tactical memory bank lives in `../agent_notes/key_learnings.md`.
- Loose session chronology lives in `../agent_notes/loose_notes/`.
- WorldFoam paper math now has a polished appendix at
  `../research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md`. It promotes
  the optical-transfer monoid, compiled cell-path atlas, same-representation
  replay theorem, and owner-run VJP, while keeping Hessians, boundary flux,
  flux witness scores, feature-gauge transfer, and ray-space transfer behind
  tests. The first paper-math implementation gate is now green at
  `../research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py`
  with tests in
  `../research_experiments/world_foam_lane2/test_cell_path_optical_transfer_fixture.py`
  and saved summary
  `../outputs/benchmarks/2026-07-08_worldfoam_cell_path_optical_transfer_summary.json`.
  It checks the optical-transfer monoid, constant-run alpha equivalence,
  same-representation replay, analytic VJP versus finite differences for
  beta/m/DeltaTau/sigma/color/run length, and the commutator swap probe
  (`render=0.0`, `element=0.0`, `grad=2.4557592070983958e-11`). The code-level
  implementation plan remains
  `../research_notes/worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md`;
  the owner-run/Metal bridge row is now green at
  `../outputs/benchmarks/2026-07-08_worldfoam_owner_run_metal_comparison_report/summary.json`.
  That bridge proves the math contract and Metal visual-capacity lane are both
  present; it is still not a full optical-transfer parity proof inside the
  Metal shader. The first scoped paper-quality benchmark table is now green at
  `../outputs/benchmarks/2026-07-08_paper_quality_benchmark_table/summary.json`;
  the first Neural3D `coffee_martini` train2/holdout1 protocol is now green at
  `../outputs/benchmarks/2026-07-11_coffee_martini_train2_holdout1_protocol/summary.json`.
  It fixes train cameras to `cam04`/`cam09`, holds out `cam06`, verifies the
  LLFF calibration path, and records separate train/heldout metrics for World
  Tubes, dynamic 3DGS, and WorldFoam. The matched 128px/16f/40-step/
  1024-primitive sweep for seeds 17/29/43 is complete at
  `../outputs/benchmarks/2026-07-11_coffee_martini_matched_sweep/report/summary.json`.
  All scoped gates pass, including offline W&B media and the promotable
  deterministic World Tubes policy. World Tubes wins mean heldout PSNR
  (`6.3863`) over clean WorldFoam (`5.6311`) and dynamic 3DGS (`4.9544`). Next
  paper work is camera-triplet and scene breadth, not reopening this split's
  runner plumbing; native WorldFoam shader parity remains conditional on the
  broader quality table.
- World Tubes now has the first executable paper-runner spine at
  `../research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py`
  with verifier coverage in
  `../tests/test_star_uvt_projective_decisive_demo_report.py`. The saved
  fixture artifact is
  `../outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/summary.json`:
  per-frame replay and one compiled interval atlas match exactly
  (`max_image_abs_error_vs_reference=0.0`, `psnr_vs_reference=120.0`) while the
  compiled atlas uses `0.125x` interval entries and `0.216x` payload memory.
  The first visibility stress suite is also green at
  `../research_experiments/star_uvt_feature_tubes/projective_visibility_stress_suite.py`
  with verifier coverage in
  `../tests/test_star_uvt_projective_visibility_stress_suite.py` and saved
  artifact
  `../outputs/benchmarks/2026-07-08_star_uvt_projective_visibility_stress_suite/summary.json`.
  It records stable clean order, raw crossing collapse, stratified crossing
  repair, and forced fallback collapse. The decisive-demo artifact now also has
  a saved real-video media row from the 128px/16f/2048-tube visual compare:
  `real_video_min_psnr=21.768529415130615`,
  `real_video_max_l1=0.054596319794654846`, and five report-side media
  artifacts under
  `../outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/real_video_media/`.
  The first scoped paper-quality benchmark table now compares World Tubes,
  WorldFoam, and dynamic 3DGS on the shared 128px/16-frame local-video
  capacity tier; the next step is scaling that table to paper datasets,
  repeats, and heldout/novel-view splits.
- The shared paper-runner table surface now exists at
  `../research_experiments/paper_runner_suite/paper_runner_table_report.py`
  with verifier coverage in `../tests/test_paper_runner_table_report.py` and
  saved report
  `../outputs/benchmarks/2026-07-11_paper_runner_table_report/summary.json`.
  It consumes the two World Tubes fixture reports, the WorldFoam
  optical-transfer fixture, the WorldFoam owner-run/Metal comparison report,
  the scoped local-video quality table, the matched three-seed `coffee_martini`
  heldout-camera table, and the 128px capacity visual-compare report. It has
  nine green evidence rows, no missing IDs, and `paper_ready=true` for the
  current runner spine. This is a reproducible one-split real-data ablation
  surface, not a multi-scene SOTA claim.
- The first tiny three-lane visual compare gate lives at
  `../outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_report.md`.
  It verifies local visual artifacts for dynamic WorldFoam/PowerFoam Metal,
  WorldTubes/STAR UVT Metal, and base dynamic 3DGS fast-mac Metal. The
  `tokengs` / fast-mac lane now has direct disk media output too, and the clean
  all-three summary is
  `../outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_clean_all_lanes.json`.
  The 128px medium tier is also green at
  `../outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_medium_128_report.md`,
  with all three Metal-backed lanes producing local visual artifacts. The next
  comparison step is no longer more harness plumbing: the 128px capacity tier
  is green at
  `../outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_capacity_128_report.md`.
  Next work should be representation-specific quality: STAR can scale
  tubes/steps, WorldFoam needs a quality bridge beyond color-only fixed
  geometry, and dynamic 3DGS needs better initialization/camera/loss scheduling.
- A first browser WebGPU dynamic-splat trainer lives at
  `../web/dynaworld_browser_trainer/`. It serves as a SPA from repo root,
  preloads the local Neural3D `coffee_martini` preview when available, falls
  back to a deterministic D-NeRF-style fixture, runs WGSL SGD on dynamic
  splat/tube parameters, and renders the current result live. The current UI
  defaults to 768 splats, exposes 96-768 splats, and has two approximation
  modes: World Tubes-style shared motion and dynamic-splats-style velocity. The
  latest in-app smoke on the Apple WebGPU adapter loaded `converge25`, cropped
  the side-by-side Neural3D preview to a single 128x128 source-view pane, found
  `2018` moving frame/pixel samples, defaulted Motion Mix to `95%`, and reported
  no browser warnings/errors. The current math has a static mean
  background, fixed-order source-over dynamic residual splats, aspect-aware
  distances, temporal-support control defaulting to `0.30`,
  train/render-matched 3-sigma Gaussian support, deterministic sparse
  grid/motion validation metrics, and a configurable motion/uniform WGSL
  training sampler now defaulting to 95% motion samples. The key convergence diagnosis is
  that the old global loss was misleading: initial `Grid Loss` was only
  `0.000186`, while `Motion Loss` was `0.044978`. A short hyperparameter sweep
  found LR `0.90` materially better than the old `0.45`, so `converge18`
  changes the default and fixes a reset/ResizeObserver half-initialized render
  race. `converge19` adds motion-aware initialization by seeding the last 38% of
  splats from the high-motion frame/pixel set. `converge20` changes the visible
  Motion Loss to direct loss over that packed high-motion sample set, so its
  values are not numerically comparable with older grid-weighted motion-loss
  entries. `converge21` adds a real `Steps/s` stat, separate from render-loop
  FPS, and aligns the JS trainer fallback learning-rate default with the UI
  default `0.90`. Latest verified 384-splat runs with the true motion-sample metric:
  Dynamic splats-style reached step `520` / motion loss `0.007004`; World
  Tubes-style reached step `304` / motion loss `0.007175`; both had no new
  console warnings/errors. The `converge21` Dynamic splats-style throughput
  smoke reached step `158` with `Steps/s 16.7`, `Grid Loss 0.000164`, true
  `Motion Loss 0.007612`, and no warnings/errors. The follow-up capacity sweep
  found 512 splats is the better default: 384 reached true `Motion Loss
  0.006904` by step `648`, 768 reached `0.006711` by step `339` at only
  `7.0-7.9` steps/s, and 512 reached `0.006685` by step `553` at about
  `11.6-13.5` steps/s. The post-edit `converge22` smoke loaded the new 512
  default and reached true `Motion Loss 0.007406` by step `174` with `Steps/s
  12.0`, no warnings/errors. The `converge23` temporal sweep promotes support
  `0.30`: `0.18`, `0.22`, and matched `0.26` reached true motion losses
  `0.007093`, `0.006991`, and `0.006877`, while `0.30` reached `0.006591` by
  step `574`, no warnings/errors. The post-edit `converge23` smoke loaded 512
  splats / temporal `0.30` by default and reached true `Motion Loss 0.007114`
  by step `237` with no warnings/errors. `converge24` adds the `Motion Mix`
  slider; the follow-up probe rejects LR decay and 128 samples/step, then
  promotes 95% motion sampling as `converge25` after Dynamic splats-style
  reached true `Motion Loss 0.006320` by step `831` with no warnings/errors.
  The post-edit `converge25` smoke reached true `Motion Loss 0.006886` by step
  `263` with no warnings/errors. `converge26` retests capacity under the new
  sampler and promotes 768 splats: 768 reached true `Motion Loss 0.006201` by
  step `522`, while the matched 512 rerun reached `0.006505` by step `565`.
  `converge27` adds motion-coverage and active-splat diagnostics, exposing that
  the old path lowered motion loss while dropping coverage from about `39.8%`
  to `29.9%`. `converge28` improves the motion-aware initializer, seeding 48%
  of splats from motion samples with broader/more opaque support; it reaches
  true `Motion Loss 0.005459` by step `854` while keeping motion coverage
  around `38%`, with no browser warnings/errors.
  `converge29`/`30` adds peak motion alpha plus mean opacity/radius diagnostics
  and a desktop rail scroll/cache bump, so the next convergence read can
  separate splat shrink/fade from true representation limits. `converge31`
  adds a small motion-coverage hinge in the browser train shader, but the first
  50%/0.20 setting over-preserves support and slows motion-loss improvement.
  `converge32` weakens it to a late 44%/0.08 guard: only motion-sampled pixels
  below 44% dynamic alpha coverage receive an extra support-preserving
  alpha/radius/center/time gradient. The extended `converge32` trace reached
  step `861`, true `Motion Loss 0.005914`, and motion coverage `47.0%`, versus
  `converge28` step `854` / `0.005459` / `38.2%`; keep this as the current
  support-health default unless visual inspection rejects the broader support.
  `converge33` renames the UI selector from `Shader Mode` to `Motion Model` and
  keeps World Tubes-style as the default after the current guard made it
  effectively tie Dynamic splats-style (`0.005938` / `47.4%` by step `629` vs
  `0.005914` / `47.0%` by step `861`). `converge34` is a visual/debugging
  pass: equal-width target/render panes plus an amplified motion-residual target
  view. A post-edit World Tubes-style trace reached step `268`, true
  `Motion Loss 0.006505`, and motion coverage `50.2%` with no browser
  warnings/errors. `converge38`/`39` adds result-side dynamic-layer and
  alpha-support views; the diagnostic shows support lands on the moving person
  but remains too broad across the background. `converge39` lowers the temporal
  gate floor (`sigma*0.70 -> sigma*0.30`), improving boot motion loss
  `0.011522 -> 0.011099` and reaching step `72`, true
  `Motion Loss 0.007788`, motion coverage `53.9%`. `converge40` adds
  `Static Cov`, a low-motion alpha penalty, and global opacity decay; the first
  `0.055` decay row is too aggressive because it falls to `42.4%` motion
  coverage by step `239`. `converge41` lowers decay to `0.025` and reaches step
  `294`, true `Motion Loss 0.006751`, motion coverage `44.6%`, `Static Cov
  2.6%`, and `Active 406/768`; `converge42` keeps those train constants and
  thins static-coverage validation to reduce readback cost. `converge43` adds
  a dedicated low-motion sample buffer and an 8% static sample reserve, because
  the v42 static penalty was otherwise sampled mostly through the 5% uniform
  tail. The first v43 in-app trace loaded `Static Px 16384` and reached step
  `259`, true `Motion Loss 0.006803`, motion coverage `45.5%`, `Static Cov
  2.6%`, and `Active 420/768` with no browser warnings/errors. `converge44`
  exposes the reserve as `Static Mix`; `0%` recovers the v42-style sampler,
  while the default `8%` makes the displayed effective motion mix `92%`. The
  in-app v44 smoke loaded the new assets, stepped once, and had no browser
  warnings/errors. The matched v44 control then found the static reserve is not
  the convergence culprit: `Static Mix 0%` reached step `274`, true
  `Motion Loss 0.006794`, and motion coverage `45.0%`, while default `8%`
  reached step `271`, true `Motion Loss 0.006822`, and motion coverage
  `45.3%`. `converge45` exposes the hidden support target as `Support Guard`
  and defaults it to `52%`; the first v45 trace reached step `297`, true
  `Motion Loss 0.007060`, motion coverage `48.2%`, `Static Cov 2.7%`, and
  `Active 406/768`. `converge46` adds frame-motion centroid velocity
  initialization for motion-seeded splats and reaches step `290`, true
  `Motion Loss 0.007036`, motion coverage `47.0%`, `Static Cov 2.8%`, and
  `Active 407/768`; it is a small fit win but not a support win. `converge47`
  replaces the global velocity with a local residual-match velocity and reaches
  step `279`, true `Motion Loss 0.006885`, motion coverage `48.1%`,
  `Static Cov 2.7%`, and `Active 414/768`. `converge48` adds default looping
  preview time plus a source/target camera-strip drawn from the side-by-side
  Neural3D preview; this is visual context/debugging only and does not make the
  browser trainer a target-camera or heldout-view trainer. `converge49` adds
  sparse-grid MAE/PSNR/global-luma SSIM readouts and a throttled source-view
  validation-error heat map; SSIM is validation-only, not a training loss.
  Next work should be
  real renderer/init parity:
  shared-backward/tape accumulation, tile/depth/alpha compositing,
  camera/heldout data, and an exported-training bundle contract.
- STAR UVT support/binner binfix lives in
  `../agent_notes/loose_notes/2026-05-26_18-52-25_star_uvt_binner_binfix_train.md`;
  current artifacts are
  `../outputs/benchmarks/2026-05-26_star_uvt_support_target_patch_diagnostic_binfix.md`,
  `../outputs/benchmarks/2026-05-26_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_binfix_50step_media.json`,
  and
  `../outputs/benchmarks/2026-05-26_star_uvt_support_target_patch_diagnostic_targetarea2_binfix_train.md`.
  Dense transfer is now measured at
  `../outputs/benchmarks/2026-05-26_star_uvt_birthsplit_targetarea2_binfix_dense_support.md`:
  normal/forced/oracle PSNR `7.269/14.736/21.439`, alpha `>0.1` `75.4%`, and
  best raw-opacity-bias PSNR only `8.039`. The binner fix helps dense support
  versus the pre-binfix repair row. Prefix tape is now measured at
  `../outputs/benchmarks/2026-05-26_star_uvt_targetarea2_binfix_visibility_prefix_tape.md`:
  selected tubes are absent on `0.0%`, prefix-hidden on only `1.6%`, top
  contributor on `95.7%`, and carry `93.1%` weight share over selected target
  rays. The prefix-alpha follow-up is measured at
  `../outputs/benchmarks/2026-05-28_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_binfix_prefixalpha085w2_50step_media.json`:
  it passes fixed-bin and moves selected weight `0.4114 -> 0.4419`, but dense
  support stays essentially flat at
  `../outputs/benchmarks/2026-05-28_star_uvt_birthsplit_targetarea2_binfix_prefixalpha085w2_50step_dense_support.md`
  (`7.262/14.732/21.438`, alpha `>0.1` `75.4%`). The next task is broader
  ownership/coverage or a different support sampling distribution, not another
  local support-target loss, hidden-support debug pass, or alpha-pressure-only
  repeat.
- STAR UVT gauged/projective compiled-adjoint replacement gap close lives in
  `../agent_notes/loose_notes/2026-05-25_20-13-56_compiled_adjoint_replacement_gap_close.md`;
  generated acceptance report lives at
  `../outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json`
  and the updated gap report lives at
  `../outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json`.
- STAR UVT gauged/projective final completion audit lives at
  `../outputs/benchmarks/2026-05-25_star_uvt_projective_goal_final_completion_audit/summary.json`.
  It verifies the goal/meta/key-math memory contract, ten theory subfolders,
  current progress/gap inputs, Metal forward/backward evidence, broad real-video
  acceptance, compiled-adjoint replacement, and sublinear world-side work.
- STAR UVT thread closeout lives in
  `../agent_notes/loose_notes/2026-05-17_17-01-49_star_uvt_thread_closeout.md`.
- STAR UVT hidden32 manual VJP gate lives in
  `../agent_notes/loose_notes/2026-05-19_21-30-08_star_uvt_hidden32_vjp_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_hidden32_from1500.md`.
- STAR UVT matched 512px native handoff gate lives in
  `../agent_notes/loose_notes/2026-05-19_21-39-45_star_uvt_native_handoff_512_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_native_handoff_matched_512_gate.md`.
- STAR UVT native-prep handoff gate lives in
  `../agent_notes/loose_notes/2026-05-19_21-56-20_star_uvt_native_prep_handoff_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_native_prep_handoff_gate.md`.
- STAR UVT hidden sigmoid-MSE native gate lives in
  `../agent_notes/loose_notes/2026-05-19_22-10-40_star_uvt_hidden_sigmoid_mse_native_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_hidden_sigmoid_mse_native_gate.md`.
- STAR UVT sparse hidden sigmoid-MSE native gate lives in
  `../agent_notes/loose_notes/2026-05-19_22-26-50_star_uvt_sparse_hidden_native_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_sigmoid_mse_native_gate.md`.
- STAR UVT native hidden sparse visual trainer gate lives in
  `../agent_notes/loose_notes/2026-05-19_22-38-58_star_uvt_nativehidden_trainer_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_nativehidden_trainer_gate.md`.
- STAR UVT native target-area full-cell visual VJP gate lives in
  `../agent_notes/loose_notes/2026-05-19_23-01-19_star_uvt_target_area_native_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_native_gate.md`.
- STAR UVT native target-area hidden32 follow-up gate lives in
  `../agent_notes/loose_notes/2026-05-19_23-13-11_star_uvt_native_target_area_hidden32_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden32_gate.md`.
- STAR UVT native target-area geometry/feature split diagnostic lives in
  `../agent_notes/loose_notes/2026-05-19_23-33-40_star_uvt_native_target_area_geometrysplit.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_native_target_area_geometrysplit_gate.md`.
- STAR UVT native target-area recompute-floor diagnostic lives in
  `../agent_notes/loose_notes/2026-05-19_23-40-20_star_uvt_native_target_area_recompute_floor.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_native_target_area_recompute_floor_gate.md`.
- STAR UVT native target-area traversal-floor diagnostic lives in
  `../agent_notes/loose_notes/2026-05-19_23-44-21_star_uvt_native_target_area_traversal_floor.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_native_target_area_traversal_floor_gate.md`.
- STAR UVT native target-area hidden forward/backward split lives in
  `../agent_notes/loose_notes/2026-05-19_23-48-07_star_uvt_native_target_area_hidden_forward_backward_split.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden_forward_backward_split_gate.md`.
- STAR UVT native target-area hidden preact/W^T split lives in
  `../agent_notes/loose_notes/2026-05-19_23-51-58_star_uvt_native_target_area_hidden_preact_wt_split.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden_preact_wt_split_gate.md`.
- STAR UVT native target-area rowmajor W^T follow-up lives in
  `../agent_notes/loose_notes/2026-05-20_00-00-45_star_uvt_native_target_area_rowmajor_wt.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_native_target_area_rowmajor_wt_gate.md`.
- STAR UVT native target-area vec4 W^T follow-up lives in
  `../agent_notes/loose_notes/2026-05-20_00-13-26_star_uvt_native_target_area_vec4_wt.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_native_target_area_vec4_wt_gate.md`.
- STAR UVT native target-area vec4 W^T trainer A/B promotion lives in
  `../agent_notes/loose_notes/2026-05-20_00-17-30_star_uvt_vec4_wt_trainer_ab.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_native_target_area_vec4_wt_trainer_ab_gate.md`.
- STAR UVT native target-area vec4 W^T 50-step promoted-mode gate lives in
  `../agent_notes/loose_notes/2026-05-20_00-25-24_star_uvt_vec4_wt_50step_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_native_target_area_vec4_wt_50step_gate.md`.
- STAR UVT compact target-area visual route helper gate lives in
  `../agent_notes/loose_notes/2026-05-20_00-31-44_star_uvt_compact_visual_route_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_compact_target_area_visual_route_gate.md`.
- STAR UVT compact native star-only diagnostic lives in
  `../agent_notes/loose_notes/2026-05-20_00-37-06_star_uvt_compact_native_staronly_diagnostic.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_compact_native_staronly_diagnostic.md`.
- STAR UVT compact manual-hidden64 colorizer-gradient diagnostic lives in
  `../agent_notes/loose_notes/2026-05-20_00-45-14_star_uvt_compact_manualhidden64_vjp_gate.md`;
  generated comparison report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_compact_visual_vjp_gate.md`.
- STAR UVT native target-area colorizer-gradient vec4 W^T diagnostic lives in
  `../agent_notes/loose_notes/2026-05-20_00-56-26_star_uvt_native_colorizer_vec4_vjp_gate.md`;
  generated tiny parity JSON lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_vec4_wt_tiny_gate.json`,
  and the compact VJP comparison report is
  `../outputs/benchmarks/2026-05-20_star_uvt_compact_visual_vjp_gate.md`.
- STAR UVT native target-area colorizer atomic split diagnostic lives in
  `../agent_notes/loose_notes/2026-05-20_01-03-56_star_uvt_colorizer_atomic_split_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_atomic_split_gate.md`.
- STAR UVT native target-area Torch reducer prototype diagnostic lives in
  `../agent_notes/loose_notes/2026-05-20_01-09-47_star_uvt_torch_reducer_prototype_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_native_target_area_torch_reducer_prototype_gate.md`.
- STAR UVT native target-area colorizer SIMD-reduce diagnostic lives in
  `../agent_notes/loose_notes/2026-05-20_01-18-22_star_uvt_colorizer_simdreduce_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_simdreduce_gate.md`.
- STAR UVT fast-shader goal audit lives in
  `../agent_notes/loose_notes/2026-05-20_01-23-08_star_uvt_fast_shader_goal_audit.md`;
  it marks the shader diagnostic phase complete, but keeps the broader
  scale/matched-dynamic-gsplat plan open.
- STAR UVT active-goal continuation audit lives in
  `../agent_notes/loose_notes/2026-05-20_03-37-12_star_uvt_active_goal_continuation_audit.md`;
  it includes the sparse-F1 trainer hook and keeps the full goal open because
  300-video scale, full matched dynamic-gsplat ranking, feature-world-tube, and
  WorldFoam side-lane work are not proven complete.
- STAR UVT current-state and next-decision closeout lives in
  `../agent_notes/loose_notes/2026-05-20_03-46-42_star_uvt_current_state_and_next_decision.md`;
  it records what was accomplished, what remains open, and why the next useful
  goal should be a support-changing STAR UVT visibility bridge before 300-video
  scale-up.
- STAR/dynamic alpha-background ablation lives in
  `../agent_notes/loose_notes/2026-05-21_19-08-51_alpha_background_ablation.md`;
  generated summary/config/result artifacts live at
  `../outputs/benchmarks/2026-05-21_alpha_background_ablation/summary.md` and
  current-code confirmation artifacts live at
  `../outputs/benchmarks/2026-05-21_alpha_background_ablation_confirm/summary.md`.
  The latest current-code rerun lives at
  `../outputs/benchmarks/2026-05-21_alpha_background_ablation_latest/summary.md`
  and is recorded in
  `../agent_notes/loose_notes/2026-05-21_20-49-42_alpha_background_ablation_latest.md`.
  A same-code refresh lives at
  `../outputs/benchmarks/2026-05-21_alpha_background_ablation_refresh_210512/summary.md`
  and is recorded in
  `../agent_notes/loose_notes/2026-05-21_21-05-12_alpha_background_ablation_refresh.md`.
  The 100-step current-code extension lives at
  `../outputs/benchmarks/2026-05-21_alpha_background_ablation_100step_212901/summary.md`
  and is recorded in
  `../agent_notes/loose_notes/2026-05-21_21-29-01_alpha_background_100step_ablation.md`.
  It flips the short-run conclusion: post-colorizer random RGB now beats random
  feature background in both renderer families and drives dynamic-gsplat alpha
  much higher. The higher-res confirmation is now run too:
  `../outputs/benchmarks/2026-05-21_alpha_background_ablation_128px_100step_224500/summary.md`
  and
  `../outputs/benchmarks/2026-05-21_alpha_background_ablation_256px_100step_225800/summary.md`,
  recorded in
  `../agent_notes/loose_notes/2026-05-21_23-04-28_alpha_background_128_256_ablation.md`.
  The result is scale/renderer-dependent, not a universal default: dynamic
  gsplat favors post-colorizer random RGB at 256px, while STAR UVT favors
  random feature background at 256px. Use a renderer-specific setting for the
  next intended scale run; do not promote a single global background policy.
- STAR UVT CPU visibility support bridge prototype lives in
  `../agent_notes/loose_notes/2026-05-20_03-50-54_star_uvt_visibility_support_bridge_cpu_gate.md`;
  generated report/JSON live at
  `../outputs/benchmarks/2026-05-20_star_uvt_visibility_support_bridge_cpu_proxy.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_visibility_support_bridge_cpu_proxy.json`.
  It proves a support-changing geometry-gradient mechanism from a zero-hit
  target start, but it is not yet a trainer or Metal quality promotion.
- STAR UVT first-class visibility-proxy trainer gate lives in
  `../agent_notes/loose_notes/2026-05-20_04-02-52_star_uvt_visibility_proxy_trainer_gate.md`;
  generated report/JSON live at
  `../outputs/benchmarks/2026-05-20_star_uvt_visibility_proxy_trainer_gate.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_visibility_proxy_from1500_lr001_5step_media.json`.
  It passes as a mechanics gate from the sparse 1500 checkpoint with
  center/velocity gradients seen, but dense RGB remains `5.640` and proxy cost
  is about `237ms`/step, so the next gate must prove support/quality movement
  before 300-video scale-up.
- STAR UVT visibility-proxy dense-support gate lives in
  `../agent_notes/loose_notes/2026-05-20_04-10-58_star_uvt_visibility_proxy_support_gate.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-20_star_uvt_visibility_proxy_support_gate.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_visibility_proxy_dense_support_diagnostic.md`.
  It rejects the current center-only visibility proxy as the scale-up bridge:
  forced-alpha/oracle content improves, but alpha `>0.1` falls
  `41.1% -> 40.5%`, and the 10x/20-step follow-up fails trainer loss while
  reaching only `40.6%` alpha `>0.1`.
- STAR UVT opacity/precision support-aware visibility proxy lives in
  `../agent_notes/loose_notes/2026-05-20_04-20-04_star_uvt_visibility_support_proxy_gate.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-20_star_uvt_visibility_support_proxy_gate.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_visibility_support_dense_diagnostic.md`.
  It closes the missing opacity/precision-gradient plumbing in the trainer,
  but rejects this specific support objective as the next scale bridge:
  feature loss slightly worsens, dense RGB only moves `5.640 -> 5.643` versus
  the center-only row, alpha `>0.1` stays `40.5% -> 40.6%`, and proxy work
  costs `693.7ms`/step.
- STAR UVT fixed-budget visibility birth/split CPU gate lives in
  `../agent_notes/loose_notes/2026-05-20_04-30-25_star_uvt_visibility_birth_split_cpu_gate.md`;
  generated report/JSON live at
  `../outputs/benchmarks/2026-05-20_star_uvt_visibility_birth_split_cpu_gate.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_visibility_birth_split_cpu_gate.json`.
  It is the first positive mechanism gate after the support-proxy rejections:
  same-support alpha stays at `0.0` target alpha `>0.10`, the center proxy
  reaches `0.5784`, and fixed-budget birth/split reaches `1.0000` before and
  after refinement while background alpha falls `0.0479 -> 0.0072`.
  The first-class trainer opt-in now exists and passes a 512px/64f gate:
  `../agent_notes/loose_notes/2026-05-20_04-37-46_star_uvt_birthsplit_trainer_gate.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_trainer_gate.md`.
  It reallocates `32/8192` low-opacity tubes from the sparse step-1500
  checkpoint, keeps zero overflow (`100/71/128` max/p95/cap), passes 5 steps
  at `189.4ms` mean / `138.3ms` last, and lifts full RGB PSNR to `5.708`.
  The dense-support diagnostic
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_dense_support_diagnostic.md`
  says it improves normal/forced-alpha/high-alpha support versus center/support
  proxy rows (`5.708` normal, `14.606` forced-alpha, alpha `>0.5` `0.117`),
  but alpha `>0.1` only returns to `0.411` and target-background oracle falls
  to `25.234` versus `25.834` center. This is a real trainer primitive, not a
  quality promotion. The uncovered-brightness target sampler follow-up now
  lives in
  `../agent_notes/loose_notes/2026-05-20_04-50-17_star_uvt_birthsplit_uncovered_gate.md`;
  report/diagnostic live at
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_uncovered_trainer_gate.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_uncovered_dense_support_diagnostic.md`.
  It selects low-alpha bright points (`selected_alpha_mean=0.0209`) and passes
  5 steps at `187.4ms` mean with dense RGB PSNR `5.713`, but alpha `>0.1`
  remains `0.411`. The first sweep gate now lives in
  `../agent_notes/loose_notes/2026-05-20_05-02-30_star_uvt_birthsplit_sweep_gate.md`;
  reports:
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_8row.md`,
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_8row_cap256.md`,
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_n32_radius_cap128.md`.
  It shows cap `128` cannot support `64+` births without overflow, cap `256`
  clears them, and radius `96px` is the coverage lever. Best safe cap-128 row
  is `low_alpha_n32_r96_cap128`: alpha `>0.1` `0.420`, dense normal PSNR
  `5.825`, forced-alpha PSNR `14.591`, oracle `24.226`, max tile `100/128`.
  This is still not a quality promotion because oracle/content falls. Next
  implementation step was to test intermediate radii before any continuation.
  That follow-up now lives in
  `../agent_notes/loose_notes/2026-05-20_05-08-30_star_uvt_birthsplit_intermediate_radius.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_intermediate_radius_cap128.md`.
  It confirms a smooth tradeoff: uncovered `r64/r72/r80/r88` moves alpha
  `>0.1` `0.411 -> 0.413 -> 0.415 -> 0.417` while oracle falls
  `25.319 -> 25.187 -> 25.015 -> 24.802`; low-alpha `r80/r88` fails loss
  decrease with zero overflow. Next implementation step: change born-tube
  initialization, for example opacity or anisotropic support, rather than
  simply widening radius or running longer. The opacity initialization sweep
  now lives in
  `../agent_notes/loose_notes/2026-05-20_05-15-30_star_uvt_birthsplit_opacity_init_sweep.md`,
  with reports
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_opacity_init_sweep_r80_cap128.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_opacity_init_sweep_r88_cap128.md`.
  It is also a nonpromotion: lower opacity recovers oracle but gives back
  coverage, higher opacity buys only tiny coverage while lowering oracle, and
  low-alpha rows become loss-negative at higher opacity. Next implementation
  step: anisotropic birth support or another support-shape change. The
  anisotropic birth-support gate now lives in
  `../agent_notes/loose_notes/2026-05-20_05-24-30_star_uvt_anisotropic_birth_support_gate.md`,
  with reports
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_anisotropic_support_sweep_cap128.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_anisotropic_support_sweep_cap128_dense_support.md`.
  It is a clean negative: `trajectory_ellipse` rows pass with zero overflow but
  alpha `>0.1` stays `0.408-0.409`, below the prior isotropic `0.411`. Next
  implementation step: multi-center or stratified birth/split, not more
  single-line ellipse sweeps. Multi-center birth/split now lives in
  `../agent_notes/loose_notes/2026-05-20_05-30-49_star_uvt_multicenter_birth_support_gate.md`,
  with reports
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_support_sweep_cap128.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_support_sweep_cap128_dense_support.md`.
  This is a positive primitive: `farthest_xy` with `K=8`, `32` births, `r64`,
  and cap `128` reaches alpha `>0.1` `0.4309` with zero overflow and forced-alpha
  PSNR `14.608`, but oracle drops to `23.965`. Next implementation step: sweep
  multi-center `K=8` radius/opacity, not single-center shape. That sweep now
  lives in
  `../agent_notes/loose_notes/2026-05-20_05-36-45_star_uvt_multicenter_k8_radius_opacity_sweep.md`,
  with reports
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_radius_opacity_sweep_cap128.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_radius_opacity_sweep_cap128_dense_support.md`.
  Best coverage is `r72/o0.8` alpha `>0.1` `0.4318` with oracle `23.670`;
  the selected balanced row is `r64/o0.4`, alpha `>0.1` `0.4298`, forced-alpha
  `14.620`, oracle `24.805`, zero overflow, and `167.9/58.1ms` step/backward.
  The short 20-step media gate now lives in
  `../agent_notes/loose_notes/2026-05-20_05-39-52_star_uvt_multicenter_k8_r64_o04_20step_media.md`;
  config:
  `../src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_20step_media.jsonc`;
  reports:
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_media.json`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_dense_support.md`.
  It is a positive support gate: loss `0.903197 -> 0.897231`, probe PSNR
  `21.681 -> 21.769`, zero overflow, last step/backward `147.3/54.3ms`, dense
  alpha `>0.1` `0.431158`, forced-alpha `14.631`, oracle `24.851`. It is not a
  visual-quality solution. The matched `K=8/r72/o0.4` 20-step comparison now
  lives in
  `../agent_notes/loose_notes/2026-05-20_05-46-21_star_uvt_multicenter_r64_vs_r72_20step.md`;
  config:
  `../src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r72_o04_from1500_lr001_20step_media.jsonc`;
  reports:
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r72_o04_20step_media.json`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_vs_r72_o04_20step_dense_support.md`.
  It passes with zero overflow and tiny coverage/normal-PSNR gain, but gives
  back feature/probe loss and oracle (`24.851 -> 24.668`). That made
  `K=8/r64/o0.4` the balanced 20-step default; the later 50-step continuation
  and cap-safe pressure reduction are recorded below.
  The 50-step continuation config now exists at
  `../src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_50step_media.jsonc`,
  with preflight verifier
  `../research_experiments/star_uvt_feature_tubes/preflight_support_birthsplit_gate.py`
  and report
  `../outputs/benchmarks/2026-05-25_star_uvt_birthsplit_r64_o04_50step_preflight/summary.md`.
  The RGB-probe/colorizer checkpoint and V-JEPA feature cache have now been
  regenerated locally by the 1000-step probe run (`wandb` offline `onsehts5`):
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.pt`
  and
  `../outputs/feature_cache/star_uvt_feature_targets/vjepa2_1_vitb_256crop_64f/a524619cf73c9cc18bdbe53d.pt`.
  The STAR resume-ladder has also been regenerated through 1300 steps. The first
  checkpoint was rebuilt from scratch
  (`wandb` offline `alkbeo34`):
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step.pt`.
  It passes with loss `1.458365 -> 1.057802`, feature loss
  `0.999935 -> 0.812539`, RGB-probe PSNR `13.387 -> 16.104`, zero overflow,
  and mean step/backward/render `3975.9/1983.0/1438.0ms`. The 300->600 segment
  then produced
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_600step_after_resume.pt`
  with loss `1.056025 -> 0.752422`, feature loss `0.811725 -> 0.654100`,
  RGB-probe PSNR `16.121 -> 20.074`, and zero overflow. The 600->800 segment
  produced
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_800step_after_resume.pt`
  with total loss `0.556334 -> 0.403675`, RGB-probe PSNR
  `20.078 -> 22.458`, and zero overflow; feature loss increases there
  (`0.653852 -> 0.706235`) under weight `0.25`, so do not cite it as a
  feature-loss win. The 800->1000 segment then produced
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_1000step_after_resume.pt`
  with loss `0.762971 -> 0.428924`, feature loss `0.706284 -> 0.637935`,
  RGB-probe PSNR `22.465 -> 22.598`, zero overflow, and mean
  step/backward/render `3368.5/1836.8/1137.3ms` (`wandb` offline `bubca3vm`).
  The 1000->1100 segment then produced
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_1100step_after_resume.pt`
  with total loss `0.538652 -> 0.503427`, RGB-probe PSNR
  `22.602 -> 23.537`, zero overflow, and mean step/backward/render
  `3433.4/1927.0/1125.4ms` (`wandb` offline `pvv0mbwo`); feature loss
  increases `0.637887 -> 0.652565` under weight `0.5`, so do not cite it as a
  feature-loss win. The 1100->1200 segment then produced
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_1200step_after_resume.pt`
  with loss `0.740994 -> 0.600747`, feature loss `0.652525 -> 0.624458`,
  RGB-probe PSNR `23.542 -> 23.552`, zero overflow, and mean
  step/backward/render `3443.2/1934.2/1128.2ms` (`wandb` offline `08458lgu`).
  The 1200->1250 segment then produced
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_1250step_after_resume.pt`
  with total loss `0.644657 -> 0.636518`, RGB-probe PSNR
  `23.557 -> 23.817`, zero overflow, and mean step/backward/render
  `4591.2/2448.5/1450.6ms` (`wandb` offline `y0ml2jc9`); feature loss
  increases `0.624403 -> 0.627228` under weight `0.75`, so do not cite it as a
  feature-loss win. The 1250->1300 segment then produced
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_1300step_after_resume.pt`
  with loss `0.793051 -> 0.775637`, feature loss `0.627185 -> 0.618493`,
  RGB-probe PSNR `23.823 -> 24.058`, zero overflow, and mean
  step/backward/render `4806.3/2630.3/1474.5ms` (`wandb` offline `fkjzpli1`).
  The remaining sparse-forward ladder has now been regenerated too:
  1300->1400 writes
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_1400step.pt`
  with loss `0.775389 -> 0.757040`, feature loss
  `0.618394 -> 0.609855`, RGB-probe PSNR `24.342`, zero overflow, and offline
  W&B run `inu9e86f`; 1400->1450 writes
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1400_lr005sparse_1450step.pt`
  with total loss `0.756800 -> 0.756539`, RGB-probe PSNR `24.366`, zero
  overflow, but feature loss slightly worsens `0.609756 -> 0.610156`; 1450->1500
  writes
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_1500step.pt`
  with loss `0.756490 -> 0.752234`, feature loss
  `0.610136 -> 0.608145`, RGB-probe PSNR `24.434`, zero overflow, and offline
  W&B run `hlo6xs7x`. The clean preflight is now `ready`.
  The 50-step `K=8/r64/o0.4` support run then executes but fails the final
  hard gate:
  `../outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_r64_o04_50step_media.json`.
  It is `pass=false` only because tile overflow becomes nonzero
  (`277` overflowed tiles, max tile `146/128`, overflow excess refs `1233`).
  Loss/probe movement is positive: weighted loss `0.773832 -> 0.760400`,
  feature loss `0.612675 -> 0.611403`, RGB-probe loss
  `0.004029 -> 0.003725`, RGB-probe PSNR `23.948 -> 24.289`. Next support
  work did reduce cap-128 support pressure. `K=8/n16/r48/o0.4` and
  `K=8/n16/r40/o0.4` both improve endpoint losses but still fail fixed-bin by
  two tiles (max `131/128`). The current cap-safe seed is `K=8/n8/r40/o0.4`:
  `../outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_50step_media.json`
  passes with zero overflow, max tile `123/128`, loss `0.754568 -> 0.749460`,
  feature loss `0.608402 -> 0.607554`, RGB-probe PSNR `24.372 -> 24.501`, and
  dense support better than `start1500` (`6.035 -> 6.472` normal PSNR,
  `10.702 -> 14.018` forced-alpha, `16.787 -> 21.602` oracle). The longer
  safe-row gate now selects the 90-step checkpoint
  `../outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_90step_checkpointselect_media.json`:
  `pass=true`, zero overflow, max tile `122/128`, loss `0.754568 -> 0.747006`,
  feature loss `0.608402 -> 0.606764`, RGB-probe PSNR `24.372 -> 24.552`.
  The 100-step sibling stays fixed-bin but fails after late jumps at global
  steps `1590` and `1594` (`0.755682` final loss). Dense support is nearly flat
  across 50/90/100. A checkpoint-aware tail schedule now passes the 100-step
  row (`0.749454` final loss, zero overflow, max `122/128`) but remains worse
  than the selected 90-step checkpoint and has the same dense support profile
  (`6.462/14.012/21.578` normal/forced/oracle PSNR). The allocation follow-up
  is now measured too: uniform `n16`, one-tube-per-center `K=16/n16`, and
  `K=16/n16/r32` still fail cap-128 by two tiles, while cap-safe
  `K=12/n12/r40/o0.4` does not beat the selected `K=8/n8` 90-step checkpoint
  or move dense forced-alpha/oracle support. The first cap-aware bridge is now
  measured too: cap-slack target scoring selected low-load target pixels but
  still overflowed by two tiles, exact-fit tile repair drifted to one final
  overflow tile, and guarded repair (`K=16/n16/r40/o0.4`, guard `2`) passes
  fixed-bin with max `127/128`, loss `0.753847 -> 0.749102`, and dense
  normal/forced/oracle PSNR `6.486/14.021/21.571`. The residual-cap-slack scorer
  now also passes fixed-bin and improves scalar objective/probe a little
  (`0.753586 -> 0.748839`, probe `24.404 -> 24.520`), but dense support remains
  flat (`6.486/14.019/21.579`). Footprint-aware residual birth is now measured
  too: it gives the best K16 scalar row (`0.752912 -> 0.748672`, probe
  `24.420 -> 24.521`) but dense support is still flat
  (`6.481/14.021/21.576`). The first model handoff is now measured:
  target-group-mean feature init improves the K16 scalar/content row
  (`0.752454 -> 0.748504`, dense `6.488/14.054/21.629`) but leaves alpha
  coverage flat (`>0.1` `0.655`). The first support-target alpha objective
  learns locally (`0.492962 -> 0.478448`) and nudges dense support to
  `6.508/14.084/21.626`, alpha `>0.1` `0.657`, but still leaves the
  forced-alpha/oracle gap intact. The support-target-area 2x2 patch bridge is
  cheaper and learns locally (`0.597970 -> 0.581641`), but lands on the same
  dense plateau (`6.507/14.085/21.627`, alpha `>0.1` `0.657`) while weakening
  feature loss. The binner repair and prefix-tape follow-up supersede that
  route: selected support now owns sampled target rays, so the next task is a
  broadened-ownership/coverage train before any Softmax-GS STAR port
  or WorldFoam mainline switch.
- Dynamic gsplat 512px matched smoke comparator lives in
  `../agent_notes/loose_notes/2026-05-20_01-26-48_dynamic_gsplat_512_matched_probe.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_dynamic_gsplat_512_matched_probe.md`.
- Dynamic gsplat fixed-512 20-step media comparator lives in
  `../agent_notes/loose_notes/2026-05-20_03-42-22_dynamic_gsplat_fixed512_20step_media_gate.md`;
  generated report and JSON live at
  `../outputs/benchmarks/2026-05-20_dynamic_gsplat_fixed512_20step_matched_media.md`
  and
  `../outputs/benchmarks/2026-05-20_dynamic_gsplat_fixed512_20step_matched_media.json`.
- STAR UVT selected visual-quality gate lives in
  `../agent_notes/loose_notes/2026-05-20_01-31-14_star_uvt_selected_visual_quality_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_selected_visual_quality_gate.md`.
- STAR UVT trainable RGB-grid low-frequency bridge gate lives in
  `../agent_notes/loose_notes/2026-05-20_01-48-12_star_uvt_rgb_grid_lowfreq_bridge_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_rgb_grid_lowfreq_bridge_gate.md`.
- STAR UVT compact target-area plus RGB-grid bridge gate lives in
  `../agent_notes/loose_notes/2026-05-20_01-51-59_star_uvt_compact_rgbgrid_bridge_gate.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_compact_rgbgrid40_visual_bridge_gate.md`.
- STAR UVT dense alpha/coverage diagnostic lives in
  `../agent_notes/loose_notes/2026-05-20_01-58-22_star_uvt_dense_alpha_failure_diagnostic.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_failure_diagnostic.md`.
- STAR UVT alpha-to-one coverage gate lives in
  `../agent_notes/loose_notes/2026-05-20_02-09-10_star_uvt_alpha_coverage_gate.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-20_star_uvt_compact_alpha1_coverage_gate.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_alpha1_dense_alpha_diagnostic.md`.
- STAR UVT phase-covered alpha-to-one coverage gate lives in
  `../agent_notes/loose_notes/2026-05-20_02-18-39_star_uvt_phase_alpha_coverage_gate.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-20_star_uvt_phase_alpha1_coverage_gate.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_phase_alpha1_dense_alpha_diagnostic.md`.
- STAR UVT target-aware black-hole coverage gate lives in
  `../agent_notes/loose_notes/2026-05-20_02-32-16_star_uvt_blackhole_coverage_gate.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-20_star_uvt_blackhole4_coverage_gate.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_blackhole4_dense_alpha_diagnostic.md`.
- STAR UVT target-background composition gate lives in
  `../agent_notes/loose_notes/2026-05-20_02-45-46_star_uvt_target_background_composition_gate.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-20_star_uvt_target_background_composition_gate.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_targetbg_alpha1_dense_alpha_diagnostic.md`.
- STAR UVT alpha-sweep plus patch4 support gate lives in
  `../agent_notes/loose_notes/2026-05-20_03-03-18_star_uvt_patch4_support_alpha_sweep_gate.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-20_star_uvt_patch4_support_alpha_sweep_gate.md`,
  `../outputs/benchmarks/2026-05-20_star_uvt_alpha_sweep_dense_diagnostic.md`,
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_patch4_alpha_sweep_dense_diagnostic.md`.
- STAR UVT raw-opacity bias support gate lives in
  `../agent_notes/loose_notes/2026-05-20_03-20-41_star_uvt_raw_opacity_bias_gate.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-20_star_uvt_raw_opacity_bias_gate.md` and
  `../outputs/benchmarks/2026-05-20_star_uvt_raw_opacity_bias_dense_diagnostic.md`.
- STAR UVT dense alpha-only support gate lives in
  `../agent_notes/loose_notes/2026-05-20_03-39-22_star_uvt_dense_alpha_support_gate.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_support_gate.md` and
  `../outputs/benchmarks/2026-05-20_star_uvt_densealpha075_dense_diagnostic.md`.
- STAR UVT alpha-only visibility speed profile lives in
  `../agent_notes/loose_notes/2026-05-20_03-22-41_star_uvt_alpha_only_visibility_profile.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_full.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_1chunk.md`.
- STAR UVT sparse-F1 dense-alpha trainer gate lives in
  `../agent_notes/loose_notes/2026-05-20_03-34-12_star_uvt_sparsef1_dense_alpha_trainer_gate.md`;
  generated report and JSON live at
  `../outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_sparsef1_trainer_gate.md`
  and
  `../outputs/benchmarks/2026-05-20_star_uvt_feature_targetgrid_densealpha075_sparsef1_from1500_lr001_5step_media.json`.
- STAR UVT current-goal / remaining-work snapshot lives in
  `../agent_notes/loose_notes/2026-05-19_22-18-56_star_uvt_current_goal_and_remaining.md`.
- STAR UVT doc-sync/current-state snapshot lives in
  `../agent_notes/loose_notes/2026-05-19_17-59-06_star_uvt_doc_sync_and_remaining_plan.md`.
- Follow-up shader audit / fast-overfit plan lives in
  `../agent_notes/loose_notes/2026-05-17_23-19-05_shader_audit_and_fast_overfit_plan.md`.
- Gate4 WorldFoam fused-MSE affine-clear closeout lives in
  `../agent_notes/loose_notes/2026-05-18_16-47-46_gate4_fused_mse_affineclear_closeout.md`.
- STAR UVT identity/no-pre-norm decoder diagnostic lives in
  `../agent_notes/loose_notes/2026-05-19_00-50-05_star_uvt_identity_decoder_quality_diagnostic.md`.
- STAR UVT hidden-64 decoder-capacity diagnostic lives in
  `../agent_notes/loose_notes/2026-05-19_01-17-55_star_uvt_hidden64_decoder_capacity.md`.
- STAR UVT pre-norm gain-2 colorizer-init diagnostic lives in
  `../agent_notes/loose_notes/2026-05-19_01-25-14_star_uvt_prenorm_gain2_init.md`.
- STAR UVT cached-bin sidecar diagnostic lives in
  `../agent_notes/loose_notes/2026-05-19_02-00-18_star_uvt_cached_bin_sidecar.md`.
- STAR UVT cached-bin 128/256/512 direct-mode matrix lives in
  `../agent_notes/loose_notes/2026-05-19_02-17-49_star_uvt_cached_matrix_512.md`.
- STAR UVT feature-gradient-only / two-pass split diagnostic lives in
  `../agent_notes/loose_notes/2026-05-19_02-37-56_star_uvt_feature_gradient_two_pass.md`.
- STAR UVT fixedbin/tile-slot accumulator budget gate lives in
  `../agent_notes/loose_notes/2026-05-19_02-47-53_star_uvt_tile_slot_budget.md`.
- STAR UVT tile-slot reducer isolation gate lives in
  `../agent_notes/loose_notes/2026-05-19_03-00-52_star_uvt_tile_slot_reduce_isolation.md`.
- STAR UVT reduce-vec4 fast overfit gate lives in
  `../agent_notes/loose_notes/2026-05-19_03-14-36_star_uvt_reduce_vec4_fast_overfit_gate.md`.
- STAR UVT selected-shader 128/256/512 scale gate lives in
  `../agent_notes/loose_notes/2026-05-19_03-29-14_star_uvt_selected_shader_scale_gate.md`.
- STAR UVT precomputed V-JEPA bridge audit lives in
  `../agent_notes/loose_notes/2026-05-19_03-37-23_star_uvt_precomputed_vjepa_bridge_audit.md`;
  generated audit outputs live at
  `../outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md`
  and
  `../outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.json`.
- STAR UVT cached-feature target bridge smoke lives in
  `../agent_notes/loose_notes/2026-05-19_03-44-59_star_uvt_rgbpyramid_target_bridge_smoke.md`;
  generated benchmark outputs live at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_rgbpyramid_target_bridge_smoke.md`
  and
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_rgbpyramid_target_8f_64px_512t_f32_gradcache_reduce_vec4_10step.json`.
- STAR UVT real V-JEPA target bridge smoke lives in
  `../agent_notes/loose_notes/2026-05-19_03-57-38_star_uvt_real_vjepa_target_bridge_smoke.md`;
  generated benchmark outputs live at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_bridge_smoke.md`
  and
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_8f_64px_512t_f32_gradcache_reduce_vec4_10step.json`.
- STAR UVT real V-JEPA target 512px scale gate lives in the chunked follow-up
  `../agent_notes/loose_notes/2026-05-19_04-27-00_star_uvt_chunked_vjepa_target_gate.md`;
  generated benchmark outputs live at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_512_scale_gate.md`
  and
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_chunkedtarget_lr005_5step.json`.
- STAR UVT real V-JEPA cached-target-layout follow-up lives in
  `../agent_notes/loose_notes/2026-05-19_04-56-43_star_uvt_cached_chunks_vjepa_target.md`;
  generated benchmark output lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_cachedchunks_lr005_5step.json`.
  The target-cache budget report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_target_cache_budget.md`.
- STAR UVT real V-JEPA target-grid loss follow-up lives in
  `../agent_notes/loose_notes/2026-05-19_05-13-38_star_uvt_target_grid_vjepa_loss.md`;
  generated benchmark output lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_lr005_5step.json`.
  The 20-step media follow-up lives in
  `../agent_notes/loose_notes/2026-05-19_05-20-06_star_uvt_target_grid_media_gate.md`;
  generated benchmark output lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_lr005_20step_media.json`,
  with media in `../outputs/media/`.
  The RGB-aux1 visual-control probe lives in
  `../agent_notes/loose_notes/2026-05-19_05-29-19_star_uvt_target_grid_rgb_aux_probe.md`;
  generated benchmark output lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux1_lr005_20step_media.json`,
  with media in `../outputs/media/`.
  The RGB-aux10 weak negative control lives in
  `../agent_notes/loose_notes/2026-05-19_05-36-44_star_uvt_target_grid_rgb_aux10_probe.md`;
  generated benchmark output lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux10_lr005_20step_media.json`,
  with media in `../outputs/media/`.
  The 100-step RGB-aux10 schedule probe lives in
  `../agent_notes/loose_notes/2026-05-19_05-43-53_star_uvt_target_grid_rgb_aux10_100step.md`;
  generated benchmark output lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux10_lr005_100step_media.json`,
  with media in `../outputs/media/`.
  The matched RGB-warm20 negative control lives in
  `../agent_notes/loose_notes/2026-05-19_05-59-03_star_uvt_target_grid_rgbwarm20_aux10_negative.md`;
  generated benchmark output lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbwarm20_aux10_lr005_100step_media.json`,
  with media in `../outputs/media/`.
  The standalone target-grid feature-to-RGB probe lives in
  `../agent_notes/loose_notes/2026-05-19_06-17-29_star_uvt_target_grid_feature_rgb_probe.md`;
  generated benchmark output lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.json`,
  with checkpoint
  `../outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.pt`
  and media in `../outputs/media/`.
	  The frozen RGB-probe STAR integration gate lives in
	  `../agent_notes/loose_notes/2026-05-19_06-30-27_star_uvt_frozen_rgb_probe_integration.md`;
	  generated benchmark output lives at
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_20step_media.json`,
	  with probe media in `../outputs/media/`.
	  The 100-step frozen RGB-probe follow-up lives in
	  `../agent_notes/loose_notes/2026-05-19_06-37-26_star_uvt_frozen_rgb_probe_100step.md`;
	  generated benchmark output lives at
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_100step_media.json`,
	  with probe media in `../outputs/media/`.
	  The 300-step frozen RGB-probe extension lives in
	  `../agent_notes/loose_notes/2026-05-19_06-49-30_star_uvt_frozen_rgb_probe_300step.md`;
	  generated benchmark output lives at
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step_media.json`,
	  with probe media in `../outputs/media/`.
	  The STAR feature overfit checkpoint/resume gate lives in
	  `../agent_notes/loose_notes/2026-05-19_07-07-49_star_uvt_checkpoint_resume_gate.md`;
	  the runtime smoke wrote and resumed `/tmp/star_uvt_checkpoint_resume_smoke/*.pt`
	  with optimizer state and zero overflow.
	  The real 64f/512px checkpoint/resume continuation lives in
	  `../agent_notes/loose_notes/2026-05-19_07-33-56_star_uvt_resume300_from300_probe.md`;
	  the 600->800 probe-emphasis continuation lives in
	  `../agent_notes/loose_notes/2026-05-19_08-02-42_star_uvt_probe_emphasis_resume200.md`;
	  the scheduled 800->1000 balance continuation lives in
	  `../agent_notes/loose_notes/2026-05-19_08-16-30_star_uvt_scheduled_balance_resume200.md`;
	  the feature0.5/probe40 1000->1100 Pareto continuation lives in
	  `../agent_notes/loose_notes/2026-05-19_08-30-45_star_uvt_pareto_resume100.md`;
	  the 1100->1200 recover schedule continuation lives in
	  `../agent_notes/loose_notes/2026-05-19_08-42-18_star_uvt_recover_schedule_resume100.md`;
	  the short feature0.75/probe40 1200->1250 probe-recovery continuation lives
	  in
	  `../agent_notes/loose_notes/2026-05-19_08-49-21_star_uvt_probe_recovery_resume50.md`;
	  the feature1/probe40 1250->1300 both-improving continuation lives in
	  `../agent_notes/loose_notes/2026-05-19_09-07-31_star_uvt_feature1_balance_resume50.md`;
	  the feature1/probe40 1300->1400 extension lives in
	  `../agent_notes/loose_notes/2026-05-19_09-25-10_star_uvt_feature1_balance_resume100.md`;
	  the matched 1300->1400 timing repeat lives in
	  `../agent_notes/loose_notes/2026-05-19_09-39-08_star_uvt_feature1_timing_repeat.md`;
	  the whole-graph target-grid/frozen-probe profile lives in
	  `../agent_notes/loose_notes/2026-05-19_09-51-27_star_uvt_feature1_wholegraph_profile.md`;
	  the trainer end-to-end timing trace lives in
	  `../agent_notes/loose_notes/2026-05-19_10-00-33_star_uvt_feature1_trainer_trace.md`;
	  the trainer chunk-trace spike localization lives in
	  `../agent_notes/loose_notes/2026-05-19_10-12-24_star_uvt_feature1_chunktrace.md`;
	  the optimizer/LR checkpoint gate lives in
	  `../agent_notes/loose_notes/2026-05-19_10-28-36_star_uvt_feature1_lr_checkpoint_gate.md`;
	  the effective-lr001 100-step continuation lives in
	  `../agent_notes/loose_notes/2026-05-19_10-37-10_star_uvt_feature1_lr001_100step_continuation.md`;
	  generated benchmark outputs live at
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step_checkpoint.json`
	  and
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_resume300_from300_checkpoint_media.json`,
	  plus
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.json`,
	  and
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_checkpoint_media.json`,
	  and
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.json`,
	  and
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.json`,
	  and
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_checkpoint_media.json`,
	  and
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.json`,
	  and
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.json`,
	  and
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_timing_repeat.json`,
	  plus the whole-graph profile
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_wholegraph_profile.md`,
	  plus the trainer timing trace report
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace_report.md`,
	  plus the chunk-trace spike localization report
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace_report.md`,
	  plus the optimizer/LR checkpoint report
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_reset_report.md`,
	  plus the lr001-vs-lr005 100-step continuation report
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_100step_report.md`,
	  with checkpoints in `../outputs/checkpoints/` and resumed probe media in
	  `../outputs/media/`.
	  The regenerated comparison and bridge-audit reports live at
	  `../outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`
	  and
	  `../outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md`;
	  the continuation-chain timing/quality report lives at
	  `../outputs/benchmarks/2026-05-19_star_uvt_feature1_continuation_chain.md`.
- STAR UVT logit-handoff tile-slot reducer gate lives in
  `../agent_notes/loose_notes/2026-05-19_11-13-32_star_uvt_logit_handoff_reduce_gate.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_report.md`
  and
  `../outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_matrix_256_512_64f_32768t_f32/summary.md`.
- STAR UVT logit-handoff real-video RGB-VJP profile lives in
  `../agent_notes/loose_notes/2026-05-19_11-24-40_star_uvt_logit_handoff_rgb_vjp_profile.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile_64f512_from1300.md`
  and
  `../outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile_8f64_smoke.md`.
- STAR UVT target-grid/frozen-probe VJP bridge profile lives in
  `../agent_notes/loose_notes/2026-05-19_11-38-19_star_uvt_targetgrid_vjp_bridge_profile.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-19_star_uvt_targetgrid_vjp_bridge_profile_64f512_from1300.md`
  and
  `../outputs/benchmarks/2026-05-19_star_uvt_targetgrid_vjp_bridge_profile_64f512_from1300_smoke.md`.
- STAR UVT analytic target-grid/probe VJP follow-up lives in
  `../agent_notes/loose_notes/2026-05-19_11-47-06_star_uvt_targetgrid_analytic_vjp.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_profile_64f512_from1300_repeat5.md`,
  `../outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_profile_64f512_from1300.md`,
  `../outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_profile_64f512_from1300_smoke.md`,
  and the trainer comparison
  `../outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_trainer_report.md`.
  The trainer smoke config is
  `../src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_analyticvjp.jsonc`.
- STAR UVT dense-analytic target-grid trainer render-mode matrix lives in
  `../agent_notes/loose_notes/2026-05-19_12-08-35_star_uvt_targetgrid_render_mode_matrix.md`;
  generated reports live at
  `../outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix.md`
  with a repeat-top check at
  `../outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix_repeat_top.md`.
  It benchmarks all current trainer render modes from the same 1300-step
  checkpoint. All rows pass and match final loss/probe PSNR, but vec4/reduce
  do not win that dense-analytic target-grid/frozen-probe objective end-to-end.
  The later sparse-grid and sparse-forward gates below supersede it for the
  selected speed path. The
  `feature_direct_fixedbin` request reports `kernel_backward_mode=direct_atomic`
  and is only an eligibility/fallback contract.
- STAR UVT sparse-pixel target-grid VJP gate lives in
  `../agent_notes/loose_notes/2026-05-19_12-33-20_star_uvt_sparse_pixel_vjp.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_pixel_vjp_report.md`.
  The new trainer mode
  `feature_target.image_vjp_mode=analytic_sparse_pixels` passes parity and a
  matched 5-step 64f/512 trainer smoke from the 1300-step checkpoint. It cuts
  dense analytic no-first step `1318.0ms -> 973.7ms` by visiting only
  `65,536` sparse pixels per step (`0.390625%` of dense). This is now
  superseded by the sparse-grid gate below because this mode still materializes
  and packs a dense Torch image VJP.
- STAR UVT sparse-grid target-grid VJP gate lives in
  `../agent_notes/loose_notes/2026-05-19_12-53-15_star_uvt_sparse_grid_vjp.md`;
  generated report lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_grid_vjp_report.md`.
  The new trainer mode
  `feature_target.image_vjp_mode=analytic_sparse_grid` directly maps trilinear
  target-grid/probe gradients to sparse source pixel ids/values. It passes
  profile parity (`4.60e-08` max grad error), cuts bridge total to `760.6ms`,
  and passes the matched 5-step 64f/512 trainer smoke at `795.3ms` no-first
  step and `88.6ms` no-first backward. The sparse-grid render-mode matrix
  lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparsegrid_render_mode_trainer_matrix.md`
  and keeps `feature_direct_gradcache_reduce_vec4` selected (`730.5ms`
  no-first, `78.3ms` mean backward). This supersedes sparse-pixel packing as
  the fastest backward-only current-objective reference.
- STAR UVT sparse-forward target-grid gate is recorded in the same loose note
  and report; the forward profile lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_profile.md`, and
  the trainer JSON lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforwardvjp_64f512_from1300_5step.json`.
  The new trainer mode
  `feature_target.image_vjp_mode=analytic_sparse_grid_forward` renders only the
  target-grid support pixels, folds sparse feature values into the target grid,
  and reuses sparse-grid VJP for backward. The sparse forward profile is
  bit-exact and initially cut render `515.9ms -> 70.5ms` (`7.322x`). The
  follow-up scale matrix lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_scale_128_256_512.md`.
  It passes 128/256/512 with zero overflow, but reveals timing instability:
  no-first trainer step is `379.2ms` at 128px, `494.2ms` at 256px, and
  `973.0ms` in the sequential 512px row; the isolated 512px repeat after the
  matrix is `598.2ms` no-first and `477.6ms` last step. This is the selected
  target-grid/frozen-probe diagnostic, but not a stable hard speed baseline.
  The dedicated repeat-3 512px timing gate lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_512_repeat3_timing.md`
  and passes all rows with zero overflow; no-first step mean/min/max/stdev is
  `504.9/411.0/626.4/110.3ms`, last-step is `468.8/409.3/549.9/72.7ms`, and
  no-first backward is `142.2/114.7/174.4/30.1ms`. Use that distribution as
  the next native-shader comparison surface.
- STAR UVT batched target-grid/probe VJP is now a first-class opt-in trainer
  path. The closeout/status note is
  `../agent_notes/loose_notes/2026-05-19_17-24-58_star_uvt_batched_trainer_status.md`;
  the preflight lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batched_target_vjp_profile.md`
  and the full optimizer harness lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batched_step_benchmark.md`.
  Batched target/probe VJP preserves loss and sparse gradient packs
  (`7.45e-09` loss error, `6.55e-11` max feature grad error) and cuts isolated
  loss+VJP `38.0ms -> 4.8ms` (`7.99x`). The 5-step harness is trainable and
  zero-overflow with no-first step `173.1ms`, render `71.3ms`, batched
  loss+VJP `7.3ms`, and backward `67.4ms`. The checked trainer config
  `../src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparseforward_batchedvjp.jsonc`
  passes and the repeat-3 report
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batchedvjp_512_repeat3_timing.md`
  gives no-first step mean/min/max/stdev `179.3/159.7/215.6/31.5ms`,
  no-first backward `72.0/60.8/90.2/15.9ms`, and zero overflow. Next is longer
  overfit/quality validation and native fixedbin/tile-slot or target/probe VJP
  kernels only if they beat this distribution.
- STAR UVT batched 100-step media gate now lives at
  `../outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_64f512_from1300_100step_media.md`.
  It launches through
  `../src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-feature-512-fast`
  using the config
  `../src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_sparseforward_batchedvjp_checkpoint_media.jsonc`.
  It passes 100 steps from the 1300-step checkpoint, writes checkpoint plus
  RGB-probe contact/MP4 media, keeps zero overflow, and preserves the old
  target-grid objective movement while cutting same-checkpoint mean step
  `1690.2ms -> 399.9ms`, backward `909.6ms -> 176.9ms`, and render
  `616.7ms -> 125.2ms`. The contact sheet is valid but still blurry, so the
  next work is visual quality or a real native kernel that beats this timing.
  The same helper now also exposes `star-feature-512-visual` for the compact
  target-area visual route (`930.6ms` mean step, `6.023` full RGB on the
  current-build gate) and `star-feature-512-native-fullcell` for the promoted
  full-support native vec4 W^T baseline (`3359.2ms`, `5.732` full RGB).
  The compact native star-only diagnostic is rejected (`2265.0ms`, no
  colorizer gradients), so the next native port must preserve colorizer
  gradients and beat compact autograd. The compact manual-hidden64 diagnostic
  preserves colorizer gradients but is also rejected (`2007.4ms` mean step,
  feature/probe regression), so a Python-side hidden64 rewrite is not the
  bridge. The native colorizer-gradient vec4 W^T path now passes tiny parity
  for STAR and colorizer parameter gradients, but fails the compact trainer
  gate too (`2738.7ms` mean step, `1474.2ms` backward, same feature/probe
  regression), so returned gradients alone are not enough.
- STAR UVT effective-lr001 sparse-forward 100-step media gate now lives at
  `../agent_notes/loose_notes/2026-05-19_17-53-03_star_uvt_lr001_sparse_batched_gate.md`.
  Generated outputs live at
  `../outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_lr001_64f512_from1300_100step_media.md`
  and
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_64f512_from1300_100step_media.json`.
  It uses the same 1300-step checkpoint and effective `lr=0.001`, preserves the
  dense lr001 quality endpoint (`0.630549` feature loss, `22.034` probe PSNR),
  and cuts mean step/backward/render to `372.3/158.9/119.9ms`. It is not a
  promotion over lr005 for feature alignment, and its late timing is noisy.
- STAR UVT sparse-forward checkpoint-selection gate now lives at
  `../agent_notes/loose_notes/2026-05-19_18-04-56_star_uvt_checkpoint_select_gate.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_checkpoint_select_1400_to1450.md`.
  Matched effective-lr001 50-step continuations from the two sparse 1400
  checkpoints select the lr005-sparse state: it passes to feature loss
  `0.625976` and probe PSNR `22.010`, while the lr001-sparse state fails after a
  `1444 -> 1445` jump and ends at feature loss `0.631770` / probe PSNR
  `21.843`.
- STAR UVT selected lr005-sparse 1450->1500 media gate now lives at
  `../agent_notes/loose_notes/2026-05-19_18-13-24_star_uvt_1450_to1500_media_gate.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_lr005sparse_1450_to1500_media.md`.
  It passes with feature loss `0.625962 -> 0.625428`, probe PSNR
	  `22.010 -> 22.027`, mean `315.8ms/step`, last-20 `254.0ms/step`, zero
	  overflow, and valid but still blurry probe media. Use the 1500-step
	  checkpoint for the next continuation if staying on this lineage.
- STAR UVT autograd RGB-aux probe-init from the selected sparse 1500 checkpoint
  is recorded as a negative quality bridge in
  `../agent_notes/loose_notes/2026-05-19_18-28-12_star_uvt_probeinit_rgbaux_negative.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_autograd_rgbaux_probeinit_from1500_negative.md`.
  It improves RGB loss/PSNR (`0.272626 -> 0.259968`, `5.644 -> 5.851`), but
  worsens feature loss (`0.625418 -> 0.626799`), drops frozen-probe PSNR
  (`22.028 -> 21.879`), creates trainable-colorizer media artifacts, and costs
  `5.207s/step` (`16.5x` slower than sparse 1500). Do not promote; the next
  bridge needs rendered-feature-image distribution alignment or native VJP that
  keeps the fast sparse surface.
- STAR UVT rendered-feature sparse-pixel RGB probe from the same sparse 1500
  checkpoint now lives at
  `../agent_notes/loose_notes/2026-05-19_18-41-29_star_uvt_rendered_feature_probe.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels.md`.
  It trains only a hidden64 colorizer on actual rendered sparse pixels and
  passes the sampled loss gate (`0.168261 -> 0.099014`, sparse PSNR
  `7.740 -> 10.043`) at `241.4ms/step`, but dense full-video PSNR is only
  `6.096` and media remains sparse-streaked. This is a useful diagnostic, not a
  quality promotion.
- STAR UVT rendered-feature stratified64 RGB probe from the same sparse 1500
  checkpoint now lives at
  `../agent_notes/loose_notes/2026-05-19_18-57-04_star_uvt_stratified_rendered_feature_probe.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_stratified64.md`.
  It samples `262,144` full-resolution stratified pixels/step (`4x` the prior
  rendered-feature sparse-pixel probe) and passes sampled loss
  (`0.277860 -> 0.242981`) at `331.5ms/step`, but dense full-video PSNR is still
  only `6.132`. This rules out target-grid sampling bias as the explanation;
  colorizer-only training is not enough.
- STAR UVT sparse visual VJP from the same sparse 1500 checkpoint now lives at
  `../agent_notes/loose_notes/2026-05-19_19-06-51_star_uvt_sparse_visual_vjp_gate.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe.md`.
  It uses native sparse-pixel backward to update STAR parameters from
  full-resolution sparse RGB loss (`model_grad_seen=true`,
  `colorizer_grad_seen=false`) at `336.8ms/step`, but dense full-video PSNR is
  worse (`5.739`). This proves the missing native sparse visual-VJP bridge, not
  a quality promotion.
- STAR UVT joint sparse visual VJP from the same sparse 1500 checkpoint now
  lives at
  `../agent_notes/loose_notes/2026-05-19_19-13-21_star_uvt_joint_sparse_visual_vjp_gate.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe.md`.
  It trains STAR and the hidden64 colorizer together on the same stratified64
  sparse RGB pixels (`model_grad_seen=true`, `colorizer_grad_seen=true`) and
  improves over frozen sparse VJP (`5.739 -> 6.025` full-video PSNR), but it is
  still below the colorizer-only stratified diagnostic (`6.132`) and costs
  `729.4ms/step`. This proves the joint gradient path, not a quality promotion;
  next visual work should mix sparse visual VJP with the target-grid
  feature/probe objective.
- STAR UVT mixed target-grid/probe plus sparse visual VJP from the same sparse
  1500 checkpoint now lives at
  `../agent_notes/loose_notes/2026-05-19_19-28-54_star_uvt_targetgrid_sparsevisual_mix_gate.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_mix_from1500.md`.
  It preserves the selected target-grid feature/probe objective and improves
  sparse visual sample PSNR (`5.656 -> 6.036`) with both model and colorizer
  gradients live, but dense full-video PSNR is still only `6.024` at
  `964.0ms/step`. This is a mechanics pass and quality negative; next work
  should change the visual support/basis rather than remix sparse RGB with the
  same target-grid objective.
- STAR UVT patch2x2 sparse visual support from the same sparse 1500 checkpoint
  now lives at
  `../agent_notes/loose_notes/2026-05-19_19-45-52_star_uvt_patch_sparse_visual_gate.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patch2x2_from1500.md`.
  It keeps the same `262,144` sparse visual pixels/step but samples contiguous
  `2x2` patches on a `32x32` grid. Sparse visual sample PSNR improves to
  `6.179` and mean step falls to `619.5ms`, but feature-target loss worsens
  (`0.625418 -> 0.625532`) and dense full RGB PSNR drops to `6.000`. This is a
  negative support-basis gate; next work should use a denser visual basis such
  as downsampled dense support or compact visibility/prefix tape.
- STAR UVT patch-mean64 sparse visual basis from the same sparse 1500 checkpoint
  now lives at
  `../agent_notes/loose_notes/2026-05-19_19-56-39_star_uvt_patchmean_visual_basis_gate.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patchmean64_from1500.md`.
  It samples `1,048,576` sparse visual pixels/step (`6.25%` dense) and pools
  them into `262,144` local-mean cells. It passes and restores feature/probe
  movement (`0.625418 -> 0.625345`, probe PSNR `22.028 -> 22.045`) with dense
  full RGB PSNR back to `6.023`, but costs `1124.6ms/step` and media remains
  sparse/high-frequency. This is a mechanics-positive visual-basis gate, not a
  quality promotion; the next visual lever should be compact dense visual
  gradients or visibility/prefix tape, not another sparse support shuffle.
- STAR UVT target-area64 sparse visual basis from the same sparse 1500
  checkpoint now lives at
  `../agent_notes/loose_notes/2026-05-19_20-09-54_star_uvt_target_area_visual_basis_gate.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_from1500.md`.
  It keeps the same `1,048,576` sparse visual pixels/step and `262,144` loss
  cells but compares against true area-downsampled RGB target cells. It is
  slightly faster (`1103.1ms/step`) and improves sparse visual PSNR to `6.064`,
  but dense full RGB PSNR remains `6.023` and media is unchanged. This rejects
  selected-patch target bias as the visual-quality blocker; the next lever is
  still visibility/prefix tape or a stronger compact dense visual-gradient path.
- STAR UVT phased target-area64 sparse visual basis from the same sparse 1500
  checkpoint now lives at
  `../agent_notes/loose_notes/2026-05-19_20-22-41_star_uvt_phased_target_area_visual_gate.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_phase2x2_from1500.md`.
  It keeps the same per-step support but uses
  `pixel_source=stratified_patch_grid_phase` with `patch_phase_shape=[4,4]` to
  cycle the `2x2` patch through an `8x8` target-area cell over 16 steps. It
  passes and raises sparse visual PSNR to `6.077`, but dense full RGB PSNR
  falls to `6.019` at `1169.2ms/step`. This rejects fixed support position as
  the visual-quality blocker.
- STAR UVT full-cell8 target-area sparse visual basis from the same sparse
  1500 checkpoint now lives at
  `../agent_notes/loose_notes/2026-05-19_20-31-54_star_uvt_fullcell8_visual_gate.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_from1500.md`.
  It sends gradients through every pixel in each `8x8` target-area cell
  (`16,777,216` visual pixels/step, `262,144` loss cells). Sparse visual PSNR
  improves to `5.822`, but the run is nonpassing: feature loss worsens, probe
  PSNR falls to `21.860`, dense full RGB falls to `5.722`, and mean step is
  `7526.7ms` with `5702.6ms` in sparse visual loss construction. This rejects
  Python-side full dense support as the port-forward path; use fused
  visibility/prefix tape or fused RGB/loss/gradient work next.
- STAR UVT manual hidden64 VJP for that full-cell8 target-area gate now lives at
  `../agent_notes/loose_notes/2026-05-19_20-43-35_star_uvt_manual_hidden64_visual_vjp.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_manualvjp_from1500.md`.
  It proves the hidden64 sparse visual VJP can match autograd while reducing
  sparse visual loss construction (`5702.6ms -> 3803.6ms`) and step time
  (`7526.7ms -> 6414.0ms`). The endpoint is unchanged and still nonpassing
  (`5.722` dense RGB, worse feature/probe losses), so keep it as the parity
  scaffold for a future native fused loss/visibility path, not a promotion.
- STAR UVT star-only manual hidden64 VJP for the same full-cell8 target-area
  gate now lives at
  `../agent_notes/loose_notes/2026-05-19_20-59-30_star_uvt_staronly_hidden64_vjp.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_staronly_from1500.md`.
  It skips colorizer parameter gradients, so it passes the narrower mechanics
  gate and cuts mean step to `5801.7ms`, but dense RGB drops to `5.648` and
  sparse visual PSNR barely moves. Treat it as a lower-bound diagnostic, not a
  promoted route.
- STAR UVT fast-GELU manual hidden64 VJP now lives at
  `../agent_notes/loose_notes/2026-05-19_21-07-09_star_uvt_fastgelu_vjp_reject.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_fastgelu_from1500.md`.
  It keeps colorizer gradients but swaps exact GELU derivative for the
  sigmoid-GELU derivative in the manual VJP. It is rejected: mean step is only
  slightly lower than exact manual (`6252.1ms` vs `6414.0ms`), dense RGB stays
  `5.722`, and the profiled loss-side total is worse than exact manual.
- STAR UVT compact manual-linear VJP for the same full-cell8 target-area gate
  now lives at
  `../agent_notes/loose_notes/2026-05-19_21-17-01_star_uvt_manual_linear_vjp_gate.md`;
  report:
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_linear_from1500.md`.
  It adds a no-hidden-layer `FeatureToColor` checkpoint and
  `sparse_visual.loss_vjp_mode=manual_linear`. The standalone linear probe is
  fast but weak (`16.980` full-video PSNR). The full-cell8 trainer row passes as
  a mechanics gate and cuts mean step to `2064.4ms` with sparse visual loss
  construction `383.3ms`, but feature loss slightly worsens and dense full RGB
  remains poor at `5.668`. Keep it as a lower-complexity VJP diagnostic, not a
  promoted visual route.
- STAR UVT manual hidden64 sparse visual VJP subphase profile now lives at
  `../agent_notes/loose_notes/2026-05-19_20-50-54_star_uvt_manualvjp_subphase_profile.md`;
  split reports:
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_subphase_profile_split_fullstep.md`
  and
  `../outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_staronly_subphase_profile_fullstep.md`.
  The corrected split says target-area reduction is small (`~0.12-0.13s`
  full-step), colorizer parameter accumulation is not enough to explain the
  cost, and the largest loss-side phases are exact GELU backward
  (`~1.34-1.44s`) plus fc1 (`~0.75-0.89s`). Hidden32, matched native-handoff,
  and hidden native follow-ups now say simple dense hidden fusion is not enough.
  The native target-area gate then moves full-cell8 support into a compact native
  path: it cuts the matched star-only trainer row `5801.7 -> 3496.0ms/step` and
  survives 512px native-only where Torch hidden VJP OOMs, but dense RGB remains
  `5.648`, so it is a speed/memory promotion rather than visual-quality proof.
  The native hidden32 follow-up cuts mean step further to `2464.6ms`, but fails
  the 5-step gate (`pass=false`, probe PSNR `19.481`, full RGB `5.632`), so
  decoder shrinkage is a rejected recompute lever.
- The earlier dense 512px attempt note lives at
  `../agent_notes/loose_notes/2026-05-19_04-10-54_star_uvt_vjepa_target_512_scale_gate.md`;
  it is historical, not the current checked gate.
- STAR UVT vs Gate4 WorldFoam matched scale gate lives in
  `../agent_notes/loose_notes/2026-05-18_16-52-47_star_uvt_worldfoam_matched_scale_gate.md`.
- STAR UVT vs Gate4 WorldFoam 64px follow-up lives in
  `../agent_notes/loose_notes/2026-05-18_17-01-36_star_uvt_worldfoam_64px_scale_gate.md`.
- STAR UVT / dynamic GSplat / F32 feature renderer scaling matrix lives in
  `../agent_notes/loose_notes/2026-05-18_17-13-51_renderer_scaling_matrix_star_dynamic_feature.md`;
  the generated table now also includes STAR UVT feature first-class rows and
  lives at `../outputs/benchmarks/2026-05-18_renderer_scaling_report.md`.
- STAR UVT fast feature-shader port plan lives in
  `../research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`.
- STAR UVT feature-tube Gate 0 contract lives in
  `../agent_notes/loose_notes/2026-05-18_20-01-13_star_uvt_feature_gate0_contract.md`;
  benchmark JSONs live at
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract.json`
  and
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract_mps_repeat.json`.
- STAR UVT generalized linear handoff diagnostic lives in
  `../agent_notes/loose_notes/2026-05-18_22-51-33_star_uvt_linear_handoff_diagnostic.md`.
- STAR UVT logit handoff negative gate lives in
  `../agent_notes/loose_notes/2026-05-18_23-06-35_star_uvt_logit_handoff_negative.md`.
- STAR UVT vec4 reducer / direct-mode matrix gate lives in
  `../agent_notes/loose_notes/2026-05-18_23-29-02_star_uvt_vec4_reduce_and_matrix.md`.
- STAR UVT 512px feature scale bracket lives in
  `../agent_notes/loose_notes/2026-05-18_23-42-06_star_uvt_512_feature_scale_bracket.md`.
- STAR UVT direct feature Metal gate artifacts live at
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_tiny_parity.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_64f_128_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_directatomic_serial_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_tiny_parity.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_serial_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun2_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_fused_first3_sigmoid_mse_tiny_parity.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_fused_first3_sigmoid_mse_serial_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun4_after_fused_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_rerun2_after_fused_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_tiny_parity.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_serial_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_serial_rerun2_after_skip_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_skip_colorizer_grad_tiny_parity.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_skip_colorizer_grad_serial_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_skip_colorizer_grad_serial_rerun2_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun5_after_linear_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_logit_handoff_tiny_parity.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_logit_handoff_serial_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun6_after_logit_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_vec4_tiny_parity.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_vec4_serial_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun7_after_vec4_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_serial_rerun2_after_vec4_sequential_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_rerun3_after_vec4_sequential_64f_256_32768_f32.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_mode_matrix_128_256_64f_32768t_f32/summary.md`,
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_direct_mode_matrix_128_256_512_64f_32768t_f32_cached/summary.md`,
  and
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun3_64f_256_32768_f32.json`.
- STAR UVT feature autograd / real-video mini-overfit artifacts live at
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_overfit_4f_32px_64t_f32.json`
  and
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_video_overfit_8f_64px_512t_f32_20step.json`.
- STAR UVT feature first-class / frame-chunked artifacts live at
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_overfit_4f_32px_64t_f32_chunkparity.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_20step.json`,
  and
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_20step.json`.
- STAR UVT feature 64-frame first-class scale probes live at
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_8192t_f32_chunk4_3step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_16384t_f32_chunk4_2step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_16384t_f32_chunk4_cap256_2step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_2step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_cap256_2step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_2step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_2step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_2step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_2step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_2step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_cap256_20step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_2step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_20step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_20step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_20step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_cap256_2step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_2048t_f32_chunk2_2step.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_4096t_f32_chunk2_gradcache_2step.json`,
  and
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_2step.json`.
- STAR UVT feature current fixed-bin eligibility boundary: `64f/256px/8192t`
  is zero-overflow and eligible (max tile `80`, p95 `63`); `16384t` is near
  but over cap (overflow `736`, max `151`, p95 `123`); default `32768t`
  overflows badly (`8160`, max `274`, p95 `238`). With
  `STAR_UVT_TILE_CAPACITY=256`, `16384t` is valid, unpruned `32768t` still
  overflows `216` tiles, and `32768t/alpha>=1/72/cap256` is the current best
  passing 20-step candidate (`0.31889 -> 0.29290` loss, `5.33` PSNR,
  `1.321s/step`, `1.021s` backward, max tile `252/256`). `alpha>=1/80` and
  `alpha>=1/96` improve loss slightly but overflow late; `alpha>=1/64` is the
  conservative zero-overflow fallback.
- STAR UVT feature 512px bracket: `64f/512px/4096t/F32/chunk2` and
  `8192t/F32/chunk2` both pass under `feature_direct_gradcache` with zero
  overflow (max tiles `18` and `33`), but they are slow (`6.456s/step` and
  `7.937s/step`). This is support headroom evidence, not a usable 512px
  training row.
- STAR UVT feature first-class backward breakdown now exists:
  `research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py`
  and
  `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_repeat2.md`.
  It shows the 512px bottleneck is mostly `FeatureToColor`/loss backward
  (`77.9-83.1%` of backward) rather than the renderer alone; the renderer is
  `16.9-22.1%` at 512px and about `36%` on the 256px/32768t/cap256 split.
- STAR UVT feature no-pre-norm A/B is the first big 512px speed lever:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_no_prenorm_2step.jsonc`
  passes with zero overflow and loss decrease. It drops the 512px/8192t
  first-class row from `7.94s/step`, `4.88s` backward to `3.72s/step`,
  `1.59s` backward. The 2026-05-19 20-step media A/B also passes for both
  variants: no-pre-norm is faster (`7.37s/step`, `3.37s` backward) than
  pre-norm (`11.10s/step`, `7.07s` backward), but pre-norm ends slightly better
  (`0.31742` loss / `4.984` PSNR versus `0.32053` / `4.941`). Keep no-pre-norm
  as a fast candidate, not a quality promotion.
- STAR UVT Gate 4 same-clip quality bracket is now explicit and currently
  fails feature promotion. RGB STAR direct-atomic on
  `test_data/test_video_384_128_6fps.mp4` at 64f/512px/8192t reaches `12.44`
  PSNR after 20 steps; feature STAR tops out at `4.99` PSNR after a hidden-64
  decoder-capacity row. That row is not practical: it improves only `4.984 ->
  4.987` PSNR while slowing to `19.18s/step` and `13.77s` backward. The gain-2
  pre-norm init row is similar (`4.987` PSNR, `14.12s/step`, `8.91s` backward),
  so simple decoder init is not the bridge either. The
  identity/no-pre-norm diagnostic is the fastest feature step (`2.54s/step`,
  `1.17s` backward) but worsens quality to `4.888` PSNR, so it only closes the
  easy decoder-simplification hypothesis. See
  `outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.md`.
- STAR UVT feature `feature_direct_fixedbin` is now a checked mode/fallback
  contract, not an optimized shader yet: unpruned `32768t/cap256` records
  `mode_fallback_required=true` after `216` overflow tiles, while
  `32768t/alpha>=1/72/cap256` records zero-overflow fixedbin eligibility.
  New trainer rows record the actual kernel as
  `kernel_backward_mode=direct_atomic`; do not call this fixedbin implemented.
- STAR UVT feature `feature_direct_gradcache` is the first actual feature
  fast-backward mode: serial synthetic `64f/256px/32768t/F32` improves backward
  `485.63ms -> 471.29ms`, and the first-class
  `32768t/alpha>=1/72/cap256` row passes at `1.226s/step`, `0.973s`
  backward, zero overflow.
- STAR UVT feature-gradient atomic diagnostic: benchmark-only
  `gradcache_skip_feature_grad` intentionally zeros feature gradients and keeps
  geometry/opacity parity, dropping synthetic backward to `327.71ms` versus a
  nearby full-gradcache rerun at `592.54ms`. Use this to prioritize a real
  feature-gradient reduction/RGB-grad handoff; do not use the skip mode for
  training.
- STAR UVT feature reduced-gradient prototype:
  `feature_direct_gradcache_reduce` / `gradcache_reduce_feature_grad` is
  trainable and passes parity, but it is slower than plain gradcache on the
  target row (`523.77ms` synthetic backward versus `491.07ms` same-session
  gradcache; first-class `1.261s/step`, `1.000s` backward versus gradcache
  `1.226s/step`, `0.973s`). Keep it recorded as a negative result, not the
  default.
- STAR UVT feature vec4 reduced-gradient follow-up:
  `feature_direct_gradcache_reduce_vec4` / `gradcache_reduce_feature_grad_vec4`
  packs the reduction into `float4` SIMD groups. It passes F4/F32 parity and
  improves one synthetic direct-kernel control (`484.7ms` backward versus
  same-session gradcache `528.2ms`), but the first-class cap256 row is slower
  than gradcache and scalar reduce (`2.095s/step`, `1.413s` backward versus
  `1.807s`/`1.333s` and `1.890s`/`1.395s`). Keep it selectable for diagnostics,
  not as the default.
- STAR UVT feature fast-overfit diagnostic:
  the fresh 64f/512px/8192t no-pre-norm first-class media rerun selects
  `feature_direct_gradcache_reduce_vec4` as the current fastest feature-tube
  diagnostic (`2.491s/step`, `1.184s` backward, zero overflow, identical
  `0.32053` loss / `4.941` PSNR versus gradcache at `2.858s`/`1.327s`).
  The compact table is
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_fast_overfit_reduce_vec4_summary.md`.
  Launch it with
  `../src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-feature-512-rgbfast`.
  This is not a promoted quality baseline because RGB STAR still wins the
  same-clip source-view bracket by a wide margin.
- STAR UVT selected-shader scale gate:
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_selected_shader_scale_128_256_512.md`
  compares no-pre-norm gradcache vs reduce-vec4 at 128/256/512 for
  64f/8192t/F32. Vec4 is a real 512px first-class win (`2.858s -> 2.491s`
  step, `1.327s -> 1.184s` backward), only a small 256px win
  (`1.112s -> 1.069s` step), and a tie/slight backward loss at 128px.
  The 128px row also proves support validity is stricter at low resolution:
  cap128/default alpha, cap256/default alpha, and cap256/`alpha>=1/72` all
  overflowed; the first valid 128px 8192-tube row used cap256 plus
  `alpha>=1/32`.
- STAR UVT feature cached-bin sidecar:
  `feature_direct_gradcache_cached_bins` reuses forward tile bins in backward.
  It passes parity and cuts the same-session synthetic 64f/256px/32768t/F32
  renderer backward `1068.0ms -> 935.8ms`, but the first-class
  512px/8192t/chunk2 row ties step time and is slower on measured backward
  than plain gradcache (`16.20s/step`, `10.24s` backward versus
  `16.21s/step`, `9.68s`). Keep it as evidence that rebinning is not the main
  512px fix, not as the default.
- STAR UVT feature-gradient-only / two-pass split diagnostic:
  `gradcache_feature_grad_only` keeps only feature-gradient atomics, and
  `gradcache_two_pass_feature_grad` composes that with the existing
  geometry/opacity-only pass. Tiny F4/F32 parity passes, but naive
  split-recompute is slower than full gradcache at both checked sizes
  (`1.343s`/`1.063s` versus `0.972s`/`0.692s` at 256px, and `2.471s`/`1.613s`
  versus `2.467s`/`1.379s` at 512px). Keep these as diagnostics; the real next
  shader needs compact fixedbin/tile-slot accumulation or native VJP, not
  duplicate traversal.
- STAR UVT fixedbin/tile-slot accumulator budget:
  `tile_slot_accumulator_budget.py` uses forward bins to size the next real
  accumulator. At 64f/32768t/F32, tile-slot feature accumulation would cut
  feature-gradient write count by `128x`, but prefix recompute is too expensive
  (`39.8x` at 256px, `10.8x` at 512px). A scalar f32 contribution-weight tape
  is around `1.2GiB` at 256/512px, while a per-channel tape would be
  `37-38GiB`. Next implementation should be a compact scalar weight/prefix
  tape or native VJP, not a per-channel tape or recompute design.
- STAR UVT tile-slot reducer isolation:
  `gradcache_feature_grad_only_reduce` and
  `gradcache_feature_grad_only_reduce_vec4` expose the existing tile-slot
  reducer as a feature-only accumulator. Vec4 is a real isolated win
  (`532.8 -> 449.9ms` backward at 256px, `869.1 -> 774.8ms` at 512px), and the
  full-gradient refresh shows `gradcache_reduce_feature_grad_vec4` helps the
  512px synthetic row (`1284.2 -> 1108.0ms`). Two-pass reduce composition is
  still diagnostic only because duplicate traversal keeps it behind single-pass
  gradcache.
- STAR UVT feature narrow RGB handoff prototype:
  `fused_first3_sigmoid_mse` is benchmark-only and computes
  `alpha * sigmoid(feature[:3]) -> mean MSE` VJP inside Metal. It passes F4/F32
  parity and measures `309.31ms` synthetic backward on 64f/256px/32768t/F32,
  versus `547.58ms` for a same-session gradcache rerun and `351.58ms` for a
  same-session skip-feature-gradient diagnostic. This is not the learned
  `FeatureToColor` path; the generalized linear follow-up now passes parity but
  is slower than gradcache, so the next handoff needs a different reduction
  shape.
- STAR UVT feature generalized in-tile linear handoff:
  `direct_linear_sigmoid_mse_backward` now supports `[3,F]` colorizer weights,
  bias, sigmoid MSE, and colorizer parameter gradients. It passes F4/F32 parity
  but is slower than gradcache on the target row (`615-619ms` backward versus
  `477.5ms` same-session gradcache). The skip-colorizer-gradient diagnostic was
  noisy (`598.5-714.1ms` backward), so do not promote this version to trainer
  configs.
- STAR UVT feature image-space-prep logit handoff:
  `direct_logit_handoff_backward` passes F4/F32 parity but is also slower than
  gradcache on the target row (`595.2ms` renderer backward plus `60.2ms` prep,
  `835.6ms` total, versus same-session gradcache `529.0ms` / `693.2ms`).
  Do not promote this version to trainer configs either.
- STAR UVT feature logit-handoff tile-slot reducer:
  `logit_handoff_reduce` and `logit_handoff_reduce_vec4` combine image-space
  logit prep with the existing stable-tile feature-gradient reducers. All
  64f/32768t/F32 256px/512px rows pass parity and zero overflow. Vec4 improves
  256px backward `571.7 -> 510.6ms` and 512px backward `654.8 -> 642.3ms`;
  scalar reduce regresses 512px backward to `722.5ms`. Keep vec4 as a
  diagnostic candidate, not a trainer default.
- STAR UVT feature logit-handoff real-video RGB-VJP profile:
  `star_uvt_logit_handoff_rgb_vjp_profile.py` compares standard autograd with
  manual `logit_handoff_reduce_vec4` on linear sigmoid RGB reconstruction. The
  8f/64px smoke passes at `2.27x`; the 64f/512px/8192t 1300-checkpoint row
  passes with zero loss error, `9.43e-09` max grad error, zero overflow, and a
  small timing win (`1691.0 -> 1587.4ms`, `1.065x`). This is not evidence for
  target-grid V-JEPA MSE or hidden64 frozen-probe VJP.
- STAR UVT feature first-class scale report/media live at
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_report.md`,
  `../outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_summary.json`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_contact.png`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_side_by_side.mp4`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step_contact.png`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step_side_by_side.mp4`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step_contact.png`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step_side_by_side.mp4`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step_contact.png`,
  and
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step_side_by_side.mp4`,
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step_contact.png`,
  and
  `../outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step_side_by_side.mp4`.
- WorldFoam high-cap 24-site follow-up lives in
  `../agent_notes/loose_notes/2026-05-18_17-17-12_worldfoam_highcap_24site_gate.md`.
- Corrected WorldFoam high-cap insert regression lives in
  `../agent_notes/loose_notes/2026-05-18_17-34-42_worldfoam_highcap_insertfix_gate.md`.
- WorldFoam high-cap shellsort/local-tape follow-up lives in
  `../agent_notes/loose_notes/2026-05-18_17-46-33_worldfoam_localtape_and_nextfork.md`.
- WorldFoam inline owner-run reverse-tape follow-up lives in
  `../agent_notes/loose_notes/2026-05-18_17-58-50_worldfoam_ownerrun_reverse_tape.md`.
- WorldFoam in-kernel owner-update negative lives in
  `../agent_notes/loose_notes/2026-05-18_18-13-45_worldfoam_ownerupdate_negative.md`.
- Gate4 endpoint-record fast-path follow-up lives in
  `../agent_notes/loose_notes/2026-05-18_18-56-39_gate4_endpoint_record_fastpath.md`.
- CTO hourly code-review thread setup lives in
  `../agent_notes/loose_notes/2026-05-18_20-07-50_cto_hourly_review_thread_type.md`.

Current center of gravity:

1. For the current STAR UVT lane, keep `direct_atomic + index_add` as the
   practical 64f overfit path while deterministic compact backward remains
   blocked on load-growth/backward speed.
2. Scale the V-JEPA/static-dynamic-token lane from single-view reconstruction
   toward mixed same-view plus heldout-novel-view training.
3. Turn prepared single-camera and multicam data into source/camera-disjoint
   train/validation contracts before making quality claims.
4. Keep renderer/shader research honest by separating memory/workspace wins
   from true frame-scaling wins.
5. Keep PowerFoam post-audit work separate from the V-JEPA/token-GS lane unless
   the user explicitly asks to merge them.

Closed in the May 17 STAR UVT thread:

```text
arch=star_uvt_video_overfit now launches through src/train/train.py
direct_atomic 32768-tube/200-step high-motion 64f overfit reproduced the prior
best 256px quality row through online W&B
deterministic compact tile_pair_suffix/key_sort_segmented_metal was rerun as a
first-class probe and remains too slow to promote
512px direct_atomic multires STAR UVT now has a first-class row through
src/train/train.py with W&B 4r2x8s3c and matching prior harness quality
shader-audit timing says STAR UVT 512px/64f is backward-bound, and the 300-clip
Gaussian multires path is cache-hot with prefetch active in a 12-step profile
the Gaussian 300-clip 256->512 run was stopped after 512px promotion produced
NaNs around step 2429
```

## Active Next Steps

Use these as the default "what next" list unless the user says otherwise:

1. Turn the mixed same-view/heldout trainer bridge into benchmark evidence. The
   checked-in 10-step smoke now logs final-step media and separate
   `same_view_recon` / `heldout_view_recon` curves
   (`wandb/offline-run-20260521_222750-9yvznqiq`), but this is still an
   interface trace. Run a longer W&B-enabled benchmark before promoting
   anything to `BASELINES.md`.
2. Add a non-empty heldout/eval plan for the 1k same-view manifest or document
   which external heldout manifests are being used for a given run.
3. Promote the current multicam V-JEPA/static-dynamic-token path into a real
   benchmark contract: source/camera-disjoint manifests, smoke gates, W&B run
   ids, and `BASELINES.md` rows.
4. Use the first-class 512px multi-resolution STAR UVT config as the current
   source-view overfit row, and spend STAR speed work on high-resolution
   backward/tile-load growth rather than forward throughput.
5. For STAR UVT feature tubes, use `STAR_UVT_TILE_CAPACITY=256` plus
   `alpha>=1/72` as the current best passing 32768-tube candidate, with
   `alpha>=1/64` as the safer fallback. Use `feature_direct_gradcache` as the
   current conservative valid mode and `feature_direct_fixedbin` to enforce
   overflow fallback in reports. Use the no-pre-norm
   `feature_direct_gradcache_reduce_vec4` 512px config only as the current
   fast diagnostic. Do not describe that fast diagnostic as precomputed V-JEPA:
   the bridge audit proves it is still RGB reconstruction through
   `FeatureToColor`, while cached V-JEPA targets live in a separate
   Gaussian/token trainer family. The new opt-in `rgb_pyramid` target-feature
   smoke proves the STAR trainer can now load a cached feature target and train
   directly on `render.feature_image`; the real V-JEPA target smoke now also
   passes at 8f/64px/512t with explicit `token_grid_shape=[4,16,16]`, and the
   chunked 64f/512px/8192t V-JEPA target scale gate now passes at
   `3.74s/step` with zero overflow after moving channel adaptation before dense
   grid upsampling and streaming `[2,32,512,512]` target chunks from the
   channel-adapted token grid. The cached-chunks follow-up precomputes those
   adapted chunks once (`2048MiB`, `2.04s` load/prep) and cuts the same gate to
   `1.655s/step`, `0.770s` backward, `0.601s` render, and `0.202s` target/loss.
   The target-grid follow-up keeps only the channel-adapted `[32,32,16,16]`
   V-JEPA grid resident (`1.0MiB`) and downsamples rendered feature chunks
   before loss; it passes at `1.351s/step`, `0.705s` backward, `0.548s` render,
   and `0.041s` target/loss with loss `0.999935 -> 0.999467`. The 20-step
   media follow-up keeps feature-target loss monotonic (`0.999935 -> 0.997425`)
   at `1.451s/step`, `0.722s` backward, `0.630s` render, and `0.037s`
   target/loss, but it is not RGB quality evidence because `rgb_loss_weight=0`
   and the colorizer is not trained.
   The RGB-aux1 target-grid probe trains the colorizer and decreases both
   feature loss (`0.999935 -> 0.997336`) and RGB loss
   (`0.338171 -> 0.335263`), but RGB PSNR only moves `4.709 -> 4.746` in
   20 steps while step time rises to `2.000s`; this is not enough quality
   improvement to promote the target-grid visual path.
   RGB-aux10 barely improves RGB PSNR over aux1 (`4.750`) while slightly
   worsening feature loss (`0.997547`), so weight alone is not the visual fix.
   The 100-step aux10 row moves more clearly (`RGB PSNR 4.709 -> 5.109`,
   feature loss `0.999935 -> 0.964670`) at `1.876s/step`, so schedule length
   matters, but it is still far below the RGB STAR quality bracket. The matched
   RGB-warm20 schedule (`feature=0/rgb=20` for 20 steps, then
   `feature=1/rgb=10`) is cheaper (`1.639s/step`) but worse on final RGB PSNR
   (`5.046`) and feature loss (`0.973557`), so feature-loss-skipping warmup is
   a negative control.
   The STAR V-JEPA target route has now been compared to the existing
   Gaussian/token V-JEPA rows in
   `../outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`:
   STAR V-JEPA streaming target is `3.743s/step`, STAR V-JEPA cached-chunks
   target is `1.655s/step`, STAR V-JEPA target-grid loss is `1.351s/step`
   (`1.451s/step` for the 20-step media row, about `2.000s/step` with 20-step
   RGB aux, `1.876s/step` for 100-step aux10, and `1.639s/step` for the
	   negative RGB-warm20 row), the lr005 sparse-forward batched-VJP helper row is
	   `0.400s/step` mean / `0.263s/step` last-20, and the lr001 sparse-forward
	   rerun is `0.372s/step` mean / `0.539s/step` last-20, with the selected
	   lr005-sparse 1450->1500 media gate at `0.316s/step` mean / `0.254s/step`
	   last-20, and the negative autograd RGB-aux probe-init bridge at
		   `5.207s/step`, plus the rendered-feature sparse-pixel RGB probe at
		   `0.241s/step`, the stratified64 rendered-pixel probe at `0.332s/step`,
		   the sparse visual VJP frozen-probe gate at `0.337s/step`, and the joint
		   sparse visual VJP gate at `0.729s/step`, plus the mixed
		   target-grid/probe+sparse visual VJP gate at `0.964s/step` and the
		   patch2x2 support gate at `0.620s/step`, plus the patch-mean64
		   visual-basis gate at `1.125s/step`, plus the target-area64
		   visual-basis gate at `1.103s/step`, plus the phased target-area64
		   visual-basis gate at `1.169s/step`, plus the full-cell8
		   target-area gate at `7.527s/step` and the manual hidden64 VJP variant
			   at `6.414s/step`, plus the star-only manual hidden64 diagnostic at
			   `5.802s/step` and native full-cell target-area star-only gate at
			   `3.496s/step`, plus the compact manual-linear diagnostic at
		   `2.064s/step`, manual hidden32 at `4.298s/step`, and the matched
		   512px native handoff gate with `logit_handoff_reduce_vec4`
		   `0.386s` native backward plus `0.422s` prep,
	   selected STAR RGB fast diagnostic is `2.491s/step`, Gaussian/token
   recon-only cached conditioning is
   `3.460s/step`, and Gaussian/token prediction-side V-JEPA loss is
	   `38.621s/step` with `36.762s` backward.
	   The standalone feature-to-RGB oracle reaches `20.073` full-video PSNR at
	   `2.427ms/step`. The 20-step frozen-probe STAR integration row is cheap
	   (`1.220s/step`) but barely moves probe PSNR (`13.985 -> 14.060`); the
		   100-step follow-up moves more clearly (`13.985 -> 14.641`) at
		   `1.268s/step` and feature loss `0.970035`, but it still does not close the
		   oracle gap. The 300-step extension keeps moving (`13.985 -> 16.560`) at
		   `1.355s/step` and feature loss `0.811652`, so the objective is viable but
		   still below the standalone oracle. The checkpoint/no-media repeat matches
		   that curve at `1.268s/step`, and the resumed 300-step continuation reaches
		   probe PSNR `19.884` and feature loss `0.655366` at `1.440s/step`. This
		   nearly reaches the standalone full-video upsample number (`20.073`). The
		   probe-emphasis 600->800 continuation reaches probe PSNR `21.425` at
		   `1.512s/step`, but feature loss drifts upward (`0.655132 -> 0.703820`),
		   and the scheduled 800->1000 balance row recovers feature loss
		   (`0.703862 -> 0.643852`) while giving back a little probe PSNR
		   (`21.428 -> 21.382`). The feature0.5/probe40 1000->1100 Pareto row
		   passes and raises probe PSNR `21.384 -> 21.789` at `1.461s/step`, but
		   feature loss drifts `0.643823 -> 0.656728`. The 1100->1200 recover
		   schedule pulls feature loss back down `0.656765 -> 0.635093` at
		   `1.521s/step`, but gives back a little probe PSNR
		   (`21.792 -> 21.738`) and is nonpassing. The short feature0.75/probe40
		   1200->1250 row passes and restores probe PSNR
		   `21.740 -> 21.929` at `1.523s/step`, but feature loss rises
		   `0.635066 -> 0.638799`. The feature1/probe40 1250->1300 row is the
		   first current both-improving balance row: feature loss
		   `0.638803 -> 0.632192`, probe PSNR `21.933 -> 21.963`, zero overflow,
		   and `1.285s/step`. The 1300->1400 extension also passes and improves
		   both metrics to feature loss `0.627129` and probe PSNR `21.979`, but
		   slows to `1.690s/step` on the older dense path. The sparse-forward
		   batched-VJP 100-step helper row preserves the same movement at
		   `399.9ms/step` mean and `262.9ms/step` last-20, so the next probe should
		   improve visual quality or beat that speed surface with native VJP. A
		   matched timing repeat reproduces the slow dense row at `1.711s/step` with
		   zero overflow and `68/45/128` max/p95/cap tile count. The whole-graph profile gate now
		   splits the target-grid/frozen-probe objective and shows renderer
		   backward dominates manual backward (`81.3-81.4%`), but the isolated
		   profile does not reproduce the trainer slowdown (`1565.9ms` manual at
		   step 1250 vs `1504.0ms` at step 1300). The durable reports are
		   `../outputs/benchmarks/2026-05-19_star_uvt_feature1_continuation_chain.md`
		   and
		   `../outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_wholegraph_profile.md`.
		   The trainer trace adds per-step timings to the trainer JSON and
		   confirms the 1300-source trace is slower after dropping the first step
		   (`1850.7ms` vs `1705.3ms`), with a late objective spike at global step
		   `1318`; report:
		   `../outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace_report.md`.
		   The chunk trace localizes that spike as distributed rather than a
		   single bad chunk: `27/32` chunks worsen, frames `0-15` contribute
		   `44.5%` of the weighted-loss jump, and the higher loss persists at
		   1319. Report:
		   `../outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace_report.md`.
		   The optimizer/LR checkpoint gate was then run. The trainer now
		   re-applies config LR after loading optimizer state, because the
		   1300-step checkpoint optimizer carried `0.005`; the corrected
		   retained-optimizer `lr=0.001` row records `[0.005] -> [0.001]`,
		   removes the 1318 spike, and passes with end loss `0.884576`,
		   feature loss `0.631648`, probe PSNR `21.991`, no-first
		   `1384.4ms/step`, and `748.9ms` backward. The reset-optimizer
		   `lr=0.001` control also passes (`0.884902`, `0.631614`, `21.984`)
		   but is slower in this diagnostic. Report:
		   `../outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_reset_report.md`.
			   Quality continuation should use the 1300 checkpoint with effective
			   `lr=0.001`, not another tile-overflow check; new native
			   VJP/scalar-fixedbin speed work should beat the sparse-forward
			   batched-VJP helper before replacing it.
		   The 100-step effective-lr001 continuation from 1300 also passes with
		   media/checkpoint, reaching feature loss `0.630549`, probe PSNR
		   `22.034`, mean `1463.8ms/step`, and `778.4ms` backward. It avoids
		   the early 1318 jump but later has a smaller transient at `1377->1378`;
		   the older lr005 1300->1400 row is slower and lower probe PSNR but
		   still better on final feature loss (`0.627129`) and slightly better
		   weighted loss (`0.880751`). Report:
		   `../outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_100step_report.md`.
		   The matched effective-lr001 sparse-forward rerun preserves that dense
		   lr001 endpoint at `372.3ms/step` mean and `158.9ms` backward, but it
		   keeps the same quality tradeoff and noisy late timing. Report:
		   `../outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_lr001_64f512_from1300_100step_media.md`.
		   The first explicit optimizer-LR schedule gate (`0.001` until global
		   step `1375`, then `0.00025`) is a negative promotion result: it
		   removes the `1377->1378` jump, but a comparable jump reappears at
		   `1385->1386`; the 100-step scheduled row is worse than static lr001
		   on final weighted loss (`0.881602` vs `0.880942`), feature loss
		   (`0.630803` vs `0.630549`), probe PSNR (`22.027` vs `22.034`), and
		   timing (`1506.9ms` / `807.2ms` backward vs `1463.8ms` / `778.4ms`).
		   The late 88-step trace is diagnostic and expected to fail quality
		   pass because it stops after the spike; it confirms `26/32` chunks
		   worsen at `1385->1386` with summed weighted-loss delta `0.015248`
		   and largest frame-0 chunk delta `0.001802`. Report:
		   `../outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_schedule_report.md`.
			   Checkpoint selection is resolved in favor of the lr005-sparse
			   lineage, the 1450->1500 media gate is stable but still blurry, and
			   the autograd RGB-aux probe-init bridge from sparse 1500 is a
			   nonpromotion because it worsens feature/probe losses, shows
			   trainable-colorizer media artifacts, and costs `5.207s/step`; the
				   rendered-feature sparse-pixel probe trains on the right distribution
				   but only reaches `6.096` full-video PSNR with sparse-streaked media,
				   and the `4x` denser stratified64 rendered-pixel probe still reaches
				   only `6.132`; native sparse visual VJP now updates STAR features at
				   `0.337s/step` but is quality-negative with the frozen target-grid
				   colorizer (`5.739` full-video PSNR), joint sparse visual VJP reaches
				   `6.025` at `0.729s/step`, and the mixed target-grid/probe+sparse
				   visual VJP gate stays tied at `6.024` while slowing to
				   `0.964s/step`; the patch2x2 same-pixel support gate is faster
				   (`0.620s/step`) but drops dense RGB to `6.000`; patch-mean64
				   restores `6.023` full RGB but costs `1.125s/step`; target-area64
				   is slightly faster (`1.103s/step`) and sparse visual PSNR rises to
				   `6.064`, but dense RGB/media are unchanged; phased target-area64 raises
				   sparse visual PSNR to `6.077` but dense RGB falls to `6.019`; full-cell8
				   dense support is nonpassing at `7.527s/step`, spends `5.703s` in sparse
				   visual loss construction, and drops dense RGB to `5.722`; manual hidden64
				   VJP cuts the same row to `6.414s/step` and `3.804s` sparse visual loss
				   construction but leaves quality unchanged; star-only manual hidden64 cuts
				   further to `5.802s/step` but drops dense RGB to `5.648`; fast-GELU
				   manual hidden64 is rejected at `6.252s/step` with the same bad `5.722`
				   dense RGB; compact manual-linear cuts full-cell8 to `2.064s/step` and
				   `0.383s` sparse visual loss construction, but the weak linear decoder
				   leaves dense RGB at only `5.668`; split profiling says
					   exact GELU backward plus fc1 dominate that loss work, not target-area reduction
					   or colorizer parameter accumulation, and scalar derivative swaps do not fix it.
						   Native target-area star-only cuts the same endpoint to
						   `3.496s/step` but keeps dense RGB at `5.648`; native hidden32
						   cuts to `2.465s/step` but fails the gate with probe PSNR
						   `19.481`, so new visual work must change objective/support or
						   reduce hidden64 native reverse recompute without shrinking capacity.
						   A benchmark-only skip-feature-grad mode then shows raw
						   `grad_feature` atomics are only a small slice of hidden64 native
						   target-area backward (`594.9 -> 562.2ms` at 256px,
						   `1918.6 -> 1854.3ms` at 512px), so feature-atomic-only reducers
						   are not the next factor win. The opposite feature-only split
						   confirms simple gradient masking is not enough:
						   full/feature-only/geometry-only backward is
						   `581.3/548.2/547.3ms` at 256px and
						   `1919.7/2106.7/2174.0ms` at 512px. Recompute-only,
						   with all output-gradient atomics disabled, is still
						   `571.3ms` at 256px and `2101.7ms` at 512px, proving
						   the shared replay/hidden64 VJP envelope is the floor.
						   Traversal-only then drops to `194.9ms` and `742.2ms`,
						   so hidden64 forward/VJP is the largest removable slice
						   (`376.5ms` at 256px, `1359.6ms` at 512px). Hidden-forward-only
						   splits that slice into forward `150.6/450.6ms` and backward
						   `225.8/909.0ms` at 256/512px, making W^T/GELU-feature VJP the
						   larger subtarget. Hidden-preact-only splits that again:
						   output+GELU prebackward is only `54.8/61.7ms`, while the F32
						   W^T feature-gradient matvec is `171.0/847.3ms`,
					   while new speed work should beat the
					   sparse-forward batched-VJP helper.
			   The logit-handoff vec4 reducer gate is the current
		   speed-shader sidecar: it passes synthetic parity, improves direct
		   256/512 rows, and now passes a real-video linear RGB-VJP profile from
		   the 64f/512 1300 checkpoint. It is still not the current target-grid
		   V-JEPA/frozen-probe objective. The target-grid/frozen-probe bridge
		   profile now proves a manual image-space VJP plus direct Metal backward
		   matches autograd on that objective (`2.57e-08` max grad error). The
		   first autograd-image bridge is not faster (`1545.5ms` autograd versus
		   `1594.3ms` bridge), but the analytic target-grid/probe VJP follow-up
		   keeps parity (`3.07e-08` max grad error) and gives a small repeat-5 win
		   (`1510.6 -> 1477.2ms`). The trainer opt-in is now wired and passes a
		   matched 5-step 64f/512 gate, but it ties end-to-end step time
		   (`1303.6ms` autograd versus `1304.6ms` warm analytic rerun; no-first
		   `1264.1ms` versus `1259.2ms`). The backward bucket improves by
		   `103.3ms`, but manual VJP moves work into the loss bucket. Keep
		   analytic VJP as a diagnostic until a longer/fused gate wins real step
		   time; scalar fixedbin/tile-slot renderer work remains a separate speed
			   path. The hidden sigmoid-MSE native gate then passes correctness but
		   rejects naive dense hidden fusion as the speed route: H32 scalar is
		   `317.5/610.9/2549.4ms` total at 128/256/512px, H64 256px is
		   `817.3ms`, and vec4 reduce is slower than scalar; next native work
		   should avoid dense `[T,H,W,F]` support or use visibility/prefix tape.
		   A new alpha-only visibility profile now proves the cheap diagnostic
		   shape for scalar alpha support: existing sparse-pixel rendering with a
		   dummy F1 feature preserves alpha exactly, matches alpha-only
		   geometry/opacity gradients within `4.7e-7`, and cuts the dense-alpha
		   render+backward envelope `1100.8 -> 634.6ms` across the 32 frame chunks.
		   This should replace dense F32 alpha rendering for future alpha-only
		   diagnostics, but it does not unblock scale because the dense-alpha
		   objective itself was quality-negative. The trainer opt-in
		   `dense_alpha.render_mode=sparse_f1` reproduces the same negative endpoint
		   while cutting mean step/backward `2558.6/1114.2 -> 873.3/370.0ms` and
		   dense-alpha render/loss/backward
		   `834.5/124.6/858.9 -> 276.0/22.0/303.7ms`; use it as the cheaper
		   diagnostic route, not as a quality promotion.
		   The cache budget says the dense cached-chunks path becomes `4GiB` at
	   128f/512px/F32 or 64f/512px/F64 and `8GiB` at 64f/1024px/F32, so the next
   gate is a trained/frozen feature-to-RGB probe or a native-VJP loss that
   avoids resident multi-GiB
   targets without changing the objective silently.
   The first `feature_direct_gradcache_reduce` attempt,
   cached-bin sidecar, and default-pre-norm vec4 row are trainable/correct but
   not first-class wins, so the next shader should use a different shape. The
   new first-class
   backward split also shows that pure renderer work cannot solve 512px by
   itself: the dense `FeatureToColor`/loss VJP is `77.9-83.1%` of 512px
   backward. Next work should combine optimized fixedbin/tile-slot
   feature-gradient accumulation with compact scalar weights/prefixes plus an
   image-space colorizer/loss VJP or a
   handoff that avoids dense F32 image-gradient backprop; cap alone does not
   solve unpruned 32768 support, and the current linear/logit handoff prototypes
   are negative speed rows.
6. Do not run 512px/32768t STAR UVT feature tubes until the feature backward or
   colorize/loss path changes: 512px 4096/8192t already pass without overflow
   but cost `6.46-7.94s/step` under the default pre-norm colorizer. No-pre-norm
   and identity/no-pre-norm prove the speed direction (`7.37s/step` and
   `2.54s/step`), but both lose quality to default pre-norm. The selected
   no-pre-norm vec4 diagnostic is fastest at 512px, but the 128/256/512 scale
   gate shows it is not a universal low-res default and that 128px needs much
   stronger support pruning. Hidden-64 pre-norm and gain-2 pre-norm barely
   improve quality and are too slow. Gate 4 also shows feature STAR is far
   below RGB STAR on same-clip source overfit, so do not treat it as the default
   512px feature overfit script until the feature decoder/objective closes that
   quality gap.
7. Fix or isolate deterministic compact backward for STAR UVT. The 2026-05-18
   kernel matrix found a short-probe speed win for `direct_fixedpoint`, but prior
   longer gates went nonfinite with poor quality; treat it as a stability
   bracket, not a replacement for `direct_atomic`. Keep the `tile_pair_suffix` /
   keyed segmented path blocked until its load-growth/backward row is
   competitive again.
7. Fix or guard the Gaussian trainer's 512px promotion NaNs before trusting the
   300-clip multires config as a completed scale baseline; the 12-step 256px
   profile proves cached V-JEPA + prefetch throughput, not 512px stability.
8. Make the next WorldFoam shader gate remove endpoint-row construction from
   Python, not just candidate replay from Metal. The fused-MSE/affine-forward
   cap is now truly `256`: a Metal regression proves rows beyond `128` are
   inserted, and the 64px/24-site corrected capcheck with max row `222`
   verifies. The latest compute keeper is the lean `owner-run-fused-mse-nomid`
   recompute path (`1.13x/1.18x` total/backward scale over `8x` frames), but
   selected tape storage still grows `9.65x`. The next justified fork is packed
   endpoint owner-run delta replay for the RGB-only fused-MSE nomid path: the
   `2026-05-19_owner_run_boundary_packed_delta_probe_render16_site24_2_4_8_16_v2.json`
   probe preserves owner/count parity, recovers lengths exactly, scales storage
   `5.76x` over `8x` frames, and uses `90,220` bytes at `16f` (`0.49x` of
   current nomid CSR). Start with base/change rows plus packed
   `owner,left,right` records and length recompute; do not spend the next fork
   on `lengths_f16` or dense fixed-cap rows. The first harness implementation
   now exists as `owner-run-delta-packed-recompute-fused-mse-nomid` and the
   moving-ray regression
   `research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py`
   matches `owner-run-fused-mse-nomid` to `1.64e-7` loss and `4.10e-7` max
   site-gradient diff in the pre-test probe. It is not timing-promoted yet:
   the clean ladder was blocked by `benchmark_environment.status=contended`.
   Next run the `2/4/8/16f` train/eval ladder with
   `--require-benchmark-environment-ok`; if it passes, attack coeff
   residency/recompute because `delta_coeff_f16` dominates resident bytes. The
   coeff factorization preflight is now positive:
   `2026-05-19_owner_run_coeff_factorization_probe_render16_site24_2_4_8_16.json`
   shows boundary planes plus per-track linear ray coefficients use `30,096`
   f32 bytes versus `1,130,496` dense coeff16 bytes (`2.66%`) with zero
   dense-depth validity mismatches and `7.14e-5` max dense-depth error. The
   Metal factorized fork now exists as
   `owner-run-delta-packed-factorized-recompute-fused-mse-nomid`; it consumes
   factorized `boundary_f32 + track_ray_coeff_f32`, removes resident
   `delta_coeff_f16`, passes moving-ray parity against
   `owner-run-fused-mse-nomid`, and has a contended functional `2/4/8/16f`
   render16/site8 ladder with selected storage scaling `1.875x` over `8x`
   frames and resident coeff storage scaling `1.0x`. The regression suite now
   checks the same constant-coeff-storage invariant across `2/4/8f`, includes a
   24-site high-cap check that the factorized `boundary_f32 + track_ray_coeff_f32`
   resident storage is below `5%` of the dense `delta_coeff_f16` path, and the
   combined focused gate passes 7 tests. Train/eval rows now emit
   `train_selected_tape_schema_storage_by_key`; the factorized Metal path now
   actually consumes int16 metadata for base offsets, track-change offsets,
   change-frame rows, and change offsets. The render16/site4/2f smoke shows
   `35,946` actual schema bytes and `11,306` actual non-coeff MPS-resident
   bytes, with the old int32 metadata absent from the selected tape. This makes
   the remaining topology growth attributable before the next shader fork. The
   naive no-frame-table hypothesis is false for moving-camera owner-run tapes:
   actual track rows can skip early frames (for example `4f` site8 tracks with
       change frames `[2, 3]` instead of `[1, 2, 3]`). The follow-on Metal fork
       `owner-run-delta-packed-factorized-frameselect-recompute-fused-mse-nomid`
       passes moving-ray parity and removes sparse scan metadata with a dense
       `frame_change_index_i16` direct-select table. A site8 comparison attempt
       made it look faster at `2/4/8f`, but the frame-select artifact ended
       contended and 16f schema storage regressed (`74,046` bytes vs regular
       `67,014`) because the dense `(track, frame)` table overtook the removed
       metadata. The next fork now exists:
       `owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid`.
       It uses `track_frame_mask_i32` plus `track_change_offsets_i16` and
       rank/popcount selection in the shader. The targeted moving-ray parity test
       and framebitmask storage regression pass, the combined focused owner-run
       suite now passes 9 tests, and
       `/tmp/worldfoam_factorized_framebitmask_smoke.json` reports `status=ok`,
       schema storage `61,760` bytes, topology storage `36,624` bytes, non-coeff
       MPS resident storage `36,736` bytes, and no stale
       `track_change_offsets_i32` resident buffer. That smoke ended contended by a
       STAR UVT run, so this is not speed promotion. The comparison gate now has
       `--include-framebitmask`, aggregate candidate selection, and retry coverage
       for contaminated frame-bitmask artifacts. A live short-window attempt
       stopped before training with
       `status=preflight_failed_before_regular`, proving the gate will not emit
       bad timing rows while the machine is contended. A longer
       `2026-05-19_factorized_selector_compare_clean_site8` attempt did not
       reach the candidate rows either: two regular-factorized artifacts ended
       contended and the gate was waiting for attempt 3 preflight when
       interrupted for reflection. The gate now catches Ctrl-C/SIGTERM and writes
       `status=interrupted` for future stopped waits instead of leaving a stale
       live-looking summary. A bounded retry
       `2026-05-19_factorized_selector_compare_clean_site8_retry_interruptfix`
       reached regular factorized after clean preflights, but unrelated pytest
       and STAR UVT jobs contaminated the end snapshot before candidate rows ran.
       That exposed a gate bug, now fixed: nonzero train/eval exits with written
       JSON artifacts are loaded and retried when the artifact is benchmark-
       contaminated instead of being reported as plain train failures. A second
       retry had good regular-factorized scaling on attempt 1 but was rejected
       for end-snapshot contamination, then hit child-side start-check
       contamination before writing `out_json` on attempt 2. That edge case is
       now covered too: child exit `2` without an artifact is retryable
       start-environment contamination, and the focused comparison-gate suite
       now clears stale per-mode top-level artifact fields across retries. The
       focused comparison-gate suite passes 12 tests. A third retry
       `2026-05-19_factorized_selector_compare_clean_site8_retry3_stalefieldfix`
       got one clean regular artifact (`1.744x` total scale, `1.263x` backward
       scale, 16f storage `3.75%` of full) and a dense frame-select artifact
       with faster-looking timings, but the frame-select end snapshot was
       contaminated by unrelated pytest work. It was stopped for reflection
	       before frame-bitmask got a clean side-by-side speed artifact. Follow-up
	       work added accepted-artifact resume and `--candidate-labels`, then
	       tightened the frame-bitmask Metal selector by caching per-track mask and
	       offset loads and parallelizing selector setup across the per-frame
	       threads. The clean site8 retry
	       `2026-05-19_factorized_selector_compare_clean_site8_retry11_framebitmask_parallelsetup`
	       now promotes frame-bitmask over regular factorized: `status=ok`,
	       `best_candidate=framebitmask`, max total ratio `0.884`, max backward
	       ratio `0.886`, max schema ratio `0.973`, max topology ratio `0.922`,
	       and max non-coeff resident ratio `0.923`; total ratios for
	       `2/4/8/16f` are `0.809/0.884/0.809/0.878`. The site24/high-cap repeat
	       `2026-05-19_factorized_selector_compare_clean_site24_retry1_framebitmask_parallelsetup`
	       also passes after rejecting one contaminated frame-bitmask artifact:
	       max total ratio `0.942`, max backward ratio `0.941`, max schema ratio
	       `0.978`, max topology ratio `0.940`, and total ratios for `2/4/8/16f`
	       are `0.854/0.916/0.864/0.942`. The next TODO is the matched STAR UVT
	       speed comparison before claiming broader competitiveness. A follow-up
	       render64/site24 frame-bitmask pass widened frame-bitmask
	       `track_change_offsets` and `change_offsets` to int32 after real
	       4f/8f prefix overflows; correctness now reaches 16f, but the 8f/16f
	       rows are contaminated and the 16f artifact shows slow-owner-run tape
	       prep dominating wall time (`221.47s` endpoint sequences, `455.77s`
	       segment tape, `137.85s` baseline compaction for train). Before another
	       shader micro-fork, remove/cache that Python tape-construction cost or
	       rerun the clean render64/site24 ladder only in a quiet window. The new
	       `--experimental-selected-only-owner-run-delta-prep` flag removes the
	       baseline segment-tape accounting phases for slow-owner-run owner-run
	       delta modes and marks artifacts as selected-only; its 4f/render64
	       smoke is correctness/path evidence only because the end benchmark
	       snapshot was contended. The train/eval path also skips unused
	       owner-run `sample_meta` allocation. The follow-up exact native owner-run
	       cutwalk fixes the failed approximate boundary-transition shortcut by
	       computing midpoint owners per cut interval in C++ and applying the
	       Python transmittance threshold. Full owner-run delta packed tests now
	       pass (`8` tests, `561.792s`), and focused CPU parity now covers a
	       duplicated multiview moving-ray fixture (`2` tests, `28.622s`) so the
	       native cutwalk is checked against Python owner-run sequences under
	       view-major sample order. The matching MPS shader-output regression
	       also passes for the original and duplicated shifted multiview fixtures
	       (`2` tests, `27.706s`). The 4f path smoke drops train sequence prep
	       to `0.553s`, and the `2/4/8/16f` native-prep path ladder is `status=ok`
	       with train sequence prep `0.190/0.693/1.261/1.329s` and backward
	       medians `2.304/2.749/4.845/3.739ms`. A stricter gated `clean_retry2`
	       completed rows but ended `contended` with backward medians
	       `2.278/2.738/2.824/3.844ms`, so it is diagnostic only. `clean_retry3`
	       also started clean but ended `contended` after unrelated `ai_trader`
	       export and font-training work appeared; its row medians were
	       `2.698/22.295/3.980/6.014ms`. The STAR comparison harness now has
	       clean-artifact/live-environment gates, and both the WorldFoam ladder
	       and STAR comparison can now wait for a quiet preflight via
	       `--wait-for-benchmark-environment-ok-timeout-s`. Next: rerun the
	       WorldFoam ladder with `--require-benchmark-environment-ok` in a quiet
	       window, then run the matched STAR UVT comparison with
	       `--require-clean-worldfoam-artifact --require-benchmark-environment-ok`
	       before making broad competitiveness claims. The preferred unattended
	       command is now
	       `research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py`;
	       it defaults to render64/site24 WorldFoam, waits for a clean
	       preflight, writes a WorldFoam artifact, then runs the matched
	       64px/896-tube STAR comparison only if the WorldFoam artifact is
	       promotable. A short blocked smoke wrote
	       `2026-05-20_native_cutwalk_worldfoam_star_blocked_smoke.promotion_summary.json`
	       with `status=worldfoam_failed` before the status label was tightened
	       to `worldfoam_preflight_failed_or_contended`. Focused wait/wrapper
	       coverage now passes `12` tests, including the WorldFoam wait helper
	       and the contended-artifact wrapper label. Bounded wait attempt
	       `2026-05-20_native_cutwalk_worldfoam_star_wait_attempt1` got a clean
	       start and completed WorldFoam rows, but the end snapshot became
	       `contended`; STAR did not launch. Its diagnostic medians are
	       `2.700/2.738/3.044/5.892ms` backward and
	       `3.116/3.240/3.651/6.865ms` total for `2/4/8/16f`. The wrapper now
	       supports `--max-worldfoam-attempts`; it retries preflight timeouts or
	       end-contended WorldFoam artifacts and only runs STAR with the first
	       clean WorldFoam artifact. It also records a check-only preflight
	       environment snapshot before each WorldFoam attempt, so blocked
	       attempts preserve the current blocker evidence. Live audit
	       `2026-05-20_native_cutwalk_worldfoam_star_preflight_audit` exited
	       `worldfoam_preflight_failed_or_contended` without launching WorldFoam
	       because unrelated `ai_trader` feature-context and pytest processes
	       were contending. The wrapper now also supports `--preflight-only` for
	       intentional readiness audits; live
	       `2026-05-20_native_cutwalk_worldfoam_star_preflight_only_audit` exited
	       without launching WorldFoam and recorded active `font_maker`,
	       `ai_trader` SFT, and STAR UVT feature-overfit blockers. Focused
	       coverage is now `14` tests. The promotion summary now also includes
	       compact `worldfoam_preflight_blocking_processes` for quick handoff
	       inspection; live
	       `2026-05-20_native_cutwalk_worldfoam_star_preflight_compact_recheck`
	       recorded `font_maker` training and pytest blockers without launching
	       WorldFoam. The compact summary now filters to actual high-CPU blockers
	       when present; live
	       `2026-05-20_native_cutwalk_worldfoam_star_preflight_filtered_continue`
	       records only hot `font_maker` training and `ai_trader` export children
	       in the compact blocker list, while the full snapshot keeps idle parent
	       wrappers. Focused coverage is now `15` tests. The wrapper summary
	       contract is now less ambiguous: top-level `worldfoam_artifact` is
	       reserved for the clean promotable artifact passed to STAR, while failed
	       or diagnostic attempts are tracked under `planned_worldfoam_artifact`,
	       `worldfoam_latest_attempt_artifact`, and
	       `worldfoam_latest_written_artifact`. A post-change preflight-only audit
	       `2026-05-20_native_cutwalk_worldfoam_star_preflight_contract_resume`
	       still blocked without launching WorldFoam and proved
	       `worldfoam_artifact=null`; hot blockers were `font_maker` training and
	       `ai_trader` SFT shadow. A newer preflight-only check
	       `2026-05-20_native_cutwalk_worldfoam_star_preflight_now` was still
	       contended by `font_maker`, an `ai_trader` export child, and git add, so
	       no timing run was launched. Focused coverage is now `16` tests; the
	       added regression covers the stale-artifact case where attempt 1 writes a
	       contended diagnostic artifact and a later preflight fails before writing
	       another artifact. The STAR command field is now split the same way:
	       `planned_star_compare_command` records the audit plan, while selected
	       `star_compare_command` stays null until a clean WorldFoam artifact is
	       actually passed to STAR. Live
	       `2026-05-20_native_cutwalk_worldfoam_star_preflight_starcmd_contract`
	       proved that shape under contention without launching WorldFoam. The
	       summary shape now has
	       `summary_schema_version=worldfoam_star_native_cutwalk_gate_v2` so future
	       readers can separate old pre-contract artifacts from current summaries.
	       Live
	       `2026-05-20_native_cutwalk_worldfoam_star_preflight_schema_contract`
	       again exited `worldfoam_preflight_failed_or_contended`, kept
	       `worldfoam_artifact=null`, `star_compare_command=null`, and
	       `worldfoam_attempts=[]`, and showed the current clean-gate blocker as a
	       hot `font_maker` torch child while the `ai_trader` monitor was only
	       background load in that snapshot. Bounded full wrapper run
	       `2026-05-20_native_cutwalk_worldfoam_star_clean_wait120` waited through
	       three 120s preflight attempts but never launched WorldFoam or STAR;
	       all attempts stayed contended by external `font_maker` torch load and
	       intermittent `ai_trader` SFT children. Focused coverage is now `17`
	       tests after adding a regression for the all-preflight-failures path:
	       repeated contended preflights must not select an artifact or STAR
	       command. The final acceptance audit now lives in
	       `research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py`:
	       the wrapper can run it directly with `--verify-promotion`, records the
	       `promotion_verifier_*` fields in the summary, and returns
	       `promotion_verification_failed` if the audit rejects an otherwise
	       successful STAR comparison. Use the integrated verifier before
	       claiming the WorldFoam native-cutwalk path is competitive with STAR.
	       The verifier unit tests pass `3` tests and the current blocked
	       preflight summary correctly fails verification. Live preflight-only
	       audit
	       `2026-05-20_native_cutwalk_worldfoam_star_preflight_verifier_integration_blocked`
	       wrote schema v2 with `worldfoam_artifact=null` and
	       `star_compare_command=null`; external `font_maker` training and an
	       `ai_trader` activation RL dataset build kept the benchmark
	       environment contended, so no timing row was launched. A later
	       readiness check remained contended by `font_maker` torch and an
	       `ai_trader` pytest training child. The latest non-timing regression
	       evidence is green: wrapper/wait/STAR/verifier contract suite `22`
	       tests, native cutwalk CPU parity `2` tests, and selected
	       framebitmask native-cutwalk MPS shader-output parity `2` tests. The
	       wrapper now supports `--preflight-stability-samples`; use `3` samples
	       with a short interval for the clean gate so late local contention is
	       caught before launch. Wrapper/wait/STAR/verifier coverage is now `23`
	       tests after adding a late-contention stability regression. Live
	       `2026-05-20_native_cutwalk_stable_preflight_blocked` requested `3`
	       stable samples, stopped on the first contended sample, and launched no
	       WorldFoam or STAR timing row.
	       The strict-background retry then completed the first clean promotion
	       summary:
	       `2026-05-20_native_cutwalk_worldfoam_star_starretry.promotion_summary.json`
	       exits `status=ok`, records one clean/promotable WorldFoam attempt and
	       one clean/promotable STAR attempt, and passes the integrated verifier.
	       The selected WorldFoam artifact reports
	       `benchmark_environment.status=background`, all acceptance flags true,
	       total means `3.008/3.014/3.323/4.095ms`, backward means
	       `2.739/2.517/2.561/3.796ms`, and `1.361x/1.386x`
	       total/backward scale over `2 -> 16f`. The matched STAR comparison
	       artifact is also `background` and reports STAR medians
	       `5.003/5.943/8.092/9.794ms` total and
	       `2.629/3.411/5.083/6.768ms` backward. Treat this as the first clean
	       Gate4 speed/scale win against the matched STAR micro-gate, not as full
	       RGB-quality parity. `BASELINES.md` now has dated rows for the
	       WorldFoam and matched STAR micro-gate. Wrapper/verifier coverage now
	       passes `27` tests after adding strict-background rejection and
	       STAR-contended retry coverage.
	       The repeated-fixture 32f extension
	       `2026-05-20_native_cutwalk_worldfoam_star_repeat32_framebitmask32fix`
	       then passed the same strict wrapper/verifier after a framebitmask
	       bit-31 fix. It requests `2/4/8/16/32f` with
	       `--repeat-loaded-frames`; the 32f WorldFoam and STAR rows both record
	       `loaded_frame_count=16`, so this is synthetic speed-scaling evidence
	       only. WorldFoam medians are `2.829/3.248/4.414/4.643/6.371ms` total
	       and `2.557/2.965/4.054/4.254/6.001ms` backward, versus matched STAR
	       `5.324/6.436/7.623/9.937/13.344ms` total and
	       `2.770/3.495/4.474/6.126/9.013ms` backward. Next WorldFoam evidence
	       should move to a real longer-than-16f fixture or larger quality-linked
	       gate, not another repeated-fixture point.
	       A render96/site48 functionality smoke then caught the next real
	       correctness blocker: framebitmask base offsets reached `83695`, so the
	       old int16 metadata path failed before writing an artifact. The shader
	       and train/eval prep now keep `base_offsets_i32` for framebitmask, with
	       a regression requiring overflow beyond int16 and clean schema
	       accounting. The saved smoke
	       `2026-05-20_worldfoam_native_cutwalk_render96_site48_2f_functionality_smoke`
	       is diagnostic only because unrelated ai_trader work contended the
	       benchmark environment. The strict follow-up
	       `2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate`
	       now passes after the wrapper rejected a first live-ai-trader-export
	       attempt and promoted attempt 2. WorldFoam render96/site48 medians are
	       `3.760/4.125/4.619ms` total and `3.480/3.847/4.331ms` backward for
	       `2/4/8f`, versus matched STAR 96px/1792-tube medians
	       `5.773/7.583/9.692ms` total and `3.614/5.161/6.719ms` backward. This
	       is the current larger fused-MSE speed gate; next evidence should be a
	       real longer-than-16f fixture or quality-linked gate, not another local
	       replay micro-variant.
	       The promotion wrapper/verifier now has an explicit real-frame
	       contract for that next gate: `--worldfoam-config` forwards a custom
	       train/eval config, `--star-video-path` forwards a custom STAR video,
	       and `--require-real-loaded-frames` rejects repeated-loaded-frame
	       artifacts during final verification. The immediate checked-fixture
	       blocker is now partly removed for 32f: new DeepView config
	       `src/dataset_configs/multicam_val_deepview_03dog_128_8fps_32f.jsonc`
	       builds a real heldout-multicam 32-frame manifest, and
	       `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc`
	       drives a one-step native-cutwalk smoke that passed with
	       `loaded_frame_count=32`, no repeat flags, loss decrease, nonzero
	       gradient, and parameter update. Its result
	       `2026-05-20_worldfoam_real32_native_cutwalk_loader_smoke.json` ended
	       benchmark-contended from `MTLCompilerService`, so it is a
	       data/correctness gate, not a speed row. A true 32f promotion still
	       needs the full wrapper/STAR comparison in a quiet environment with
	       `--require-real-loaded-frames`. The refreshed dry-run summary records
	       `frame_counts=[32]` with the real input paths. The warm strict retry
	       `2026-05-20_real32_strict_mini_wrapper_settle_retry` then proved the
	       compiled 32f WorldFoam shader path is about `2.25-2.30ms` total /
	       `1.95-2.01ms` backward, but both attempts were rejected as diagnostic
	       because live `ai_trader` TOTO exports started before the post-run
	       snapshot; no STAR comparison ran. The benchmark preflight now treats
	       that periodic TOTO MPS-export monitor as a blocker even at idle CPU,
	       so the next real32 wrapper should be launched only after that screen is
	       stopped/finished or in a clean machine window. Preflight-only wrapper
	       artifact
	       `2026-05-20_real32_preflight_toto_mps_blocker_check.promotion_summary.json`
	       confirms the fail-fast path: no WorldFoam attempt and no STAR launch
	       while the idle TOTO parent chain is active. The refreshed artifact
	       carries explicit `block_reason` fields, including
	       `periodic_mps_exporter` for the idle parent chain. The promotion
	       verifier now also rejects a `require_real_loaded_frames=true` summary
	       unless it records the custom WorldFoam config and STAR video path and
	       proves the WorldFoam preflight/run commands and STAR planned/selected
	       commands used those inputs, and that STAR planned/selected commands
	       point `--worldfoam-artifact` at the selected WorldFoam artifact. It
	       also checks the selected artifacts themselves: WorldFoam `config_path`
	       must match the recorded config, and STAR `star.video_path` must match
	       the recorded video. The wrapper now records parsed `frame_counts`, and
	       real-frame verification checks
	       WorldFoam/STAR artifact rows against that exact frame set. The wrapper
	       now also refuses `--require-real-loaded-frames` at argument parse time
	       unless both `--worldfoam-config` and `--star-video-path` are supplied;
	       tests cover neither-input and one-sided-input failures. The wrapper
	       also rejects empty, non-integer, nonpositive, and duplicate
		       `--frame-counts` at parse time. The focused lane gate is `58` tests
	       passing plus scoped static checks. Native owner-run cutwalk parity now
	       also includes a synthetic non-repeated `32f` moving-ray boundary:
	       CPU cutwalk delta parity against Python owner-run sequences and MPS
	       framebitmask fused-shader output parity both pass. A direct low-level
	       MPS regression now also forces the signed frame-31 bit
	       (`track_frame_mask_i32 = -(1 << 31)`) and proves that shader path
	       changes loss/grad relative to the all-base tape. The MPS wrapper now
	       rejects framebitmask tapes whose per-track mask popcount does not
	       match the per-track change-record span. The CPU tape builder also now
	       rejects unsorted per-track change frames, because the framebitmask
	       shader uses bit-rank/popcount to map a selected frame to a change row
		       and therefore requires strictly ascending change-frame records. The
		       framebitmask helper now also rejects malformed change-offset vectors
		       directly: empty offsets, nonzero first offsets, nonmonotonic offsets,
		       and final offsets that do not match `change_frame_i32` length. The
		       MPS wrapper now has direct negative coverage for illegal mask bits
		       too: frame `0` and bit `frame_count` are rejected with popcount held
		       constant, so the tests exercise the bounds guard rather than the
		       popcount guard. The same sparse-change validation is now shared by
		       the frame-select helper, which rejects unsorted per-track frames,
		       frame-0 changes, and non-1D offset tensors before building the int16
		       rank map. The framebitmask MPS wrapper now also rejects empty
		       `change_offsets_i32` directly instead of falling through to the
		       generic offset validator with a negative inferred change count, and
		       now validates packed base/change endpoint records before entering
		       Metal: negative records, owner codes outside `site_count`, and cut
		       ids outside `boundary_count` fail in the Python wrapper instead of
		       reaching the shader. That packed endpoint-record guard is now shared
		       by the sibling non-framebitmask wrappers too: packed recompute,
		       factorized packed, factorized frameselect, packed scalar, smallrun16,
	       materialized, and framegroup16 paths now validate base/change records
	       after offset validation and before the custom op launch. A focused
	       regression corrupts prepared non-framebitmask tapes without changing
		       record lengths and proves bad base owners and bad change-cut ids are
		       rejected at the wrapper boundary. All delta direct-config paths now
		       require a prep-time `delta_packed_records_validated` marker bound to
		       the current launch contract: raw/i16/packed record tensors,
		       topology/config tensor identities and PyTorch version counters,
		       selector-flag presence, launch scalar fields, site count, and
		       runtime track/frame counts.
		       Manually assembled direct-config tapes and stale markers after record,
		       topology, config, or selector replacement/in-place mutation fail before
		       native Metal launch instead of bypassing wrapper validation. Prepared
		       tapes set the marker after CPU packed-record range validation and native
		       config tensor creation, without adding a per-step MPS-to-CPU copy. The
		       prepared factorized packed, frameselect, and framebitmask paths also
		       require that current marker. Handcrafted framebitmask shader fixtures
		       stamp the marker only after their deliberate malformed mutation, so
		       they still reach the intended deeper wrapper validation. A
		       selector-family regression now proves the marker is required across
		       raw, packed scalar/framegroup/materialized/recompute/smallrun/
		       launch-only, i16x4, i16cols, i16x3, and factorized selectors.
		       Additional stale-marker regressions now corrupt a launch-only
		       scalar, replace rowdesc buffers, and replace i16x3 owner-reduce
		       chunk-owner topology after the marker, and reject runtime
		       track/frame-count mismatches after the marker. Runtime tensor guards
		       now also reject malformed `site_rgba` and `target_rgb_track` shape,
			       dtype, device pairing, and contiguity before native launch. The
			       direct-config marker check also validates marked tape tensor dtype,
			       device, contiguity, tensor layout/rank, fixed ABI shapes,
			       flattened packed-record divisibility, selector compatibility, and
			       scalar contract consistency. Its marker payload is now
			       `delta_direct_config_v8`; scalar marker entries are type-stable
			       rather than `int(...)`-coerced, so invalid scalar types reach the
			       scalar-contract validator instead of failing or normalizing during
			       marker construction. Packed i32 direct-config prep now
			       stamps `delta_coeff_boundary_count` plus `delta_launch_*`
			       record/count scalars for every packed selector, not only launch-only
			       variants, and scalar validation now requires those scalars for all
			       i32 packed direct-config selectors. It derives
			       `delta_launch_change_count` from `change_frame_*` or
			       `change_offsets_* - 1` so factorized frameselect/framebitmask tapes
			       cannot hide stale or deleted re-stamped change counts. A manually
			       re-stamped bad tape cannot bypass the Python boundary and reach Metal
			       with malformed boundary, track-ray, rowdesc, packed-record, stale
			       launch-count, base/change record-count, missing scalar-contract, or
			       ambiguous selector payloads.
			       The owner-run delta packed module now passes `55` tests with these
			       selector/direct-config/wrapper regressions, including
			       missing scalar-contract coverage across every i32 packed selector
			       family, and the broader WorldFoam
						       wrapper/verifier gate now passes `79` tests after the guard
						       refresh, including `unchecked` benchmark-environment probes
						       blocking strict promotion instead of behaving as clean, and
						       truly quiet `ok` snapshots promoting like `background`, plus
						       the built native packed extension fixture. The verifier now
						       rejects WorldFoam artifacts with missing acceptance metadata,
						       STAR compare rejects that condition before spending a matched
						       STAR run, and the wrapper refuses to select such an artifact.
						       The quality bridge report
						       `2026-05-20_worldfoam_star_quality_bridge_native_cutwalk_microgate.json`
						       records speed competitiveness in the clean micro-gate but no
						       RGB-quality competitiveness yet: best WorldFoam train PSNR is
						       `12.248`, heldout PSNR is `12.857`, and the train gap to STAR
						       source-overfit RGB is `17.575dB`. The existing render96/site48
						       capacity candidate is included and does not improve train PSNR
						       (`9.875` best) or any overlapping primary frame
						       (`-2.55/-2.53/-2.27dB` at `2/4/8f`); it is also flagged as
						       missing the primary `16f` row, so naive render/site capacity
						       is not enough.
						       A future quality-closing capacity candidate now gets a separate
						       matched-speed-needed flag instead of inheriting the primary
						       micro-speed gate.
							       The next quality/capacity fork is now wired but not MPS-benchmarked:
								       `--site-initialization legacy_frame_pixel_mean` keeps legacy
								       support but averages each site's color only over train samples
								       from the same frame as the site, while `legacy_pixel_mean`
								       averages over all train samples at that pixel.
								       `legacy_frame_patch3_mean` keeps the same support and timing but
								       averages a same-frame 3x3 color patch. `stratified_grid`
								       remains available as a tested negative for naive image-cell
								       spread, and `stratified_pixel_mean` tests grid support plus mean
								       color. Direct initializer tests, CLI help checks, and CPU Gate4
								       compiler smokes pass. The Gate1 CPU reference at
							       render16/site9/2f gives `legacy_sparse` train/heldout PSNR
							       `11.862/12.671`, `stratified_grid` `10.419/9.692`,
								       `legacy_pixel_mean` `13.025/14.614`,
								       `legacy_frame_pixel_mean` `13.029/14.617`,
								       `legacy_frame_patch3_mean` `12.761/14.315`, and
							       `stratified_pixel_mean` `13.679/12.611`. The grid-plus-mean
							       fork is recorded as a train-overfit rejection because heldout
							       falls below the legacy sparse baseline.
							       The next quiet MPS quality run should try
							       `legacy_frame_pixel_mean` first, not plain grid spread. Use
							       `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_site_initialization_quality_bridge.json`
								       as the machine-readable handoff (`next_mps_candidate=legacy_frame_pixel_mean`,
								       `positive_candidate_count=3`, `rejected_candidate_count=2`).
							       The Gate4 affine candidate-CSR topology probe is also
							       site-init aware now; the tiny
							       `2026-05-20_gate4_affine_candidate_csr_capacity_legacy_frame_pixel_mean_render8_site4_2_4f.json`
							       artifact passes for `legacy_frame_pixel_mean` with sublinear
							       candidate/storage scaling over `2f -> 4f`.
						       The combined readiness handoff is
						       `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_candidate_readiness.json`;
						       it gates quality plus topology and keeps
						       `quality_claim=false`, `speed_claim=false` until a clean
						       MPS artifact exists. Use
							       `research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py`
							       to launch the ready candidate; it already wrote
							       `2026-05-20_worldfoam_next_mps_legacy_frame_pixel_mean_plan.json`
							       and the latest executed preflight summary
							       `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_preflight.json`
							       failed closed as `preflight_contended`.
						       That summary now exposes top-level preflight
						       status, blocker counts/reasons, and a compact
							       blocker list; the refreshed artifact reports
							       `8` blocking rows with reasons `high_cpu`,
							       `keyword:torch`, and `periodic_mps_exporter`.
						       An earlier live sample also caught an active
						       TOTO MPS export as `keyword:mps`, so the
						       exporter remains a timing-window blocker even
						       when its parent monitor is idle.
						       The launcher also supports
						       `--preflight-stability-samples`; the current
						       plan requires `3` clean samples at `5s` spacing
						       before train/eval, and the latest blocked
						       preflight stopped at sample `1/3` with
						       `preflight_stability_ok=false`.
						       The plan/preflight summaries now include
						       `result_verifier_command`, pointing at
						       `research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py`.
							       Run that verifier on the first clean
							       `legacy_frame_pixel_mean` summary before claiming
						       quality, speed, or sublinear frame scaling; it
						       correctly fails the current contended preflight
						       summary. The launcher now also supports
						       `--verify-result`; the refreshed plan summary
						       sets `verify_result=true`, so the future clean
						       execution can fail closed inside the launcher if
						       the post-run verifier rejects the artifact. It
						       also supports whole-sequence preflight retry via
						       `--preflight-retry-timeout-s` and
						       `--preflight-retry-poll-s`; unit coverage proves
						       a clean stability sequence after a dirty first
						       attempt can launch train/eval, but the live
						       retry smoke
						       `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_retrywait_smoke.json`
						       still failed closed before train/eval. Latest
						       focused verification is `py_compile` OK and
						       `33` focused tests passing. A longer strict
						       retry execution,
						       `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_final_try.json`,
						       made `11` attempts over a `180s` retry window
						       but still failed closed as `preflight_contended`,
						       completing only `1/3` stability samples and
						       producing no train/eval artifact. Verifier runs
						       on `verified_retry2`, `retrywait_smoke`, and
						       `final_try` all fail for the expected
						       contended-preflight and missing-artifact
						       reasons. A later preflight-only recheck,
						       `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_screen_blocker_recheck.json`,
						       also failed closed at sample `1/3` with high-CPU
						       `font_maker`, high-CPU `ai_trader` pytest/export
						       children, the TOTO periodic exporter screen, and
						       a `keyword:torch` queue wrapper. The latest
						       preflight-only artifact
						       `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_actionable_blockers.json`
						       adds `preflight_external_blocker_summary`: `2`
						       high-CPU external jobs, `1` torch worker, and
						       `5` periodic exporter processes, and still has
						       no train/eval artifact. The source-only native
						       variant verifier
						       `2026-05-21_worldfoam_native_variant_source_wiring.json`
						       now passes for the `fused_direct`, `fused_csr`,
						       and `fused_slab` forks, checking schema/impl,
						       dispatch-target source definitions, Python
						       wrapper, host-loaded Metal kernel names, and
						       dynamically loaded `.metal` source membership,
						       plus `MetalKernels` field declarations/
						       initializers/uses without claiming runtime
						       speed. A fresh current preflight artifact still
						       fails closed before training on external blockers;
						       `2026-05-21_worldfoam_next_mps_current_status_recheck.json`
						       also fails closed with high-CPU `font_maker` PID
						       `92641`, the `keyword:torch` queue wrapper, and the
						       TOTO exporter chain.
						       The fork wrappers now use `torch.ops.load_library`
						       for the pure `TORCH_LIBRARY` binaries, and
						       `2026-05-21_worldfoam_native_variant_import_registration.json`
						       proves normal package import registers direct
						       `11/11`, CSR `13/13`, and slab `103/103` compiled
						       schemas after rebuilding all three forked
						       extensions from source. Rebuilt MPS correctness
						       smokes pass for direct/CSR/slab power-boundary
						       counts, and the slab mixed MPS regression suite
						       passes `8` tests. Rebuilt real-ray smokes also
						       pass for direct shared real-ray replay, CSR
						       affine moving rays, slab affine VJP without
						       ownerupdate, and slab per-track ownerupdate/VJP.
						       The prior slab ownerupdate failure was the
						       unsupported default `tiled` layout; the smoke now
						       errors when `--include-ownerupdate` is used
						       without `--layout per-track`, with focused CLI
						       regression coverage. The rebuilt-native
						       smoke-bundle verifier
						       `2026-05-21_worldfoam_rebuilt_native_smoke_bundle_verifier.json`
						       passes, requiring seven valid rebuilt smoke
						       artifacts and classifying the old failed
						       tiled-ownerupdate artifact as
						       `expected_invalid_tiled_ownerupdate`. The
						       goal-state report
						       `2026-05-21_worldfoam_fork_shader_goal_state.json`
						       records `shader_fork_smoke_state_fixed=true` but
						       `objective_complete=false` and
						       `status=blocked_external_environment` because the
						       clean real32 MPS PSNR/speed/sublinear gate still
						       has no artifact. Commit/handoff scope is recorded in
						       `../research_experiments/world_foam_lane2/2026-05-21_worldfoam_fork_shader_commit_scope.md`,
						       including submodule source directories to preserve
						       and generated native outputs to exclude. Current
						       source/import/rebuilt focused verification is `51`
						       tests passing plus the `8`-test MPS slab suite. A
						       later fresh preflight-only artifact,
						       `2026-05-20_worldfoam_next_mps_legacy_pixel_mean_fresh_preflight.json`,
						       also failed closed at sample `1/3` with `8`
						       blocking rows, including high-CPU `font_maker`,
						       high-CPU `ai_trader` pytest/report children,
						       the TOTO monitor chain, and a `keyword:torch`
						       queue wrapper. Follow-up probe
						       `2026-05-20_worldfoam_next_mps_legacy_pixel_mean_probe2351.json`
						       also failed closed at sample `1/3`, catching a
						       live TOTO quote snapshot and multiple high-CPU
						       pytest/RL children. Probe
						       `2026-05-20_worldfoam_next_mps_legacy_pixel_mean_probe2354.json`
						       also failed closed at sample `1/3` with high-CPU
						       `font_maker`, high-CPU `ai_trader`
						       imitation/integrity pytest children, the TOTO
						       monitor chain, and a `keyword:torch` queue wrapper.
							       A 2026-05-21 frame-local candidate preflight,
								       `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_verified.json`,
							       also failed closed at sample `1/3`: high-CPU `font_maker`,
							       high-CPU `ai_trader` monitor/check/export children, the periodic
							       `ai_trader` MPS exporter chain, and a torch queue wrapper were
							       all active. No PSNR/speed claim exists for this fork yet.
						       Train/eval and STAR-compare benchmark capture now
						       ignore the current process ancestor chain, so an
					       `rtk sh -lc ... powerfoam_metal...` launch wrapper cannot
					       self-block as `keyword:metal`. The adjacent no-timing regression
		       slices also pass: factorized selector plus native packed/cutwalk
	       compiler tests (`24` tests) and the mixed fused-slab MPS shader suite
		       (`8` tests). A clean-evening strict wrapper attempt then got one true
		       32f WorldFoam diagnostic row:
		       `2026-05-20_real32_strict_mini_wrapper_clean_evening.attempt1.worldfoam.json`
		       records `loaded_frame_count=32`, `repeat_loaded_frames=false`,
		       `3.104ms` total, `2.773ms` backward, train PSNR `12.987`, and
		       heldout PSNR `14.229`. It is not promotable: restarted live
		       `ai_trader` offline TOTO MPS-export monitors plus transient
			       `MTLCompilerService` contaminated the post-run snapshot, attempt 2
			       stayed preflight-contended, the promotion summary ended
			       `worldfoam_preflight_failed_or_contended`, and no STAR compare ran.
			       The next timing/PSNR gate still requires pausing/stopping those TOTO
			       exporter screens or a clean machine window. The wrapper now keeps
				       the planned STAR artifact path separate from the selected
				       `star_compare_artifact`, so failed/preflight-contended summaries do
				       not look as if a STAR artifact was promoted. The verifier now
				       requires exactly one promotable STAR attempt and matching
				       selected/latest STAR artifact paths.
   Shell-sort depth replay,
   same-owner forward-merge, and in-kernel
   boundary-pair owner-update were negatives. The inline owner-run reverse-tape
   merge remains the best self-contained 24-site fork
   (`2.724/3.233/6.032/6.610 ms` total for `2/4/8/16f`, unchanged PSNR), but it
   still misses the formal scale verifier (`2.427x` total median, `2.590x`
   backward median for `8x` frames). The Gate4 endpoint-record fast path now
   proves the STAR-shaped warm kernel at 64px/24-site real frames. The latest
   direct-delta plus vectorized-coefficient artifact verifies with
   `2.303/3.594/2.255/2.967 ms` total median,
   `1.955/3.269/1.971/2.625 ms` backward median, `1.040x` storage scale, and
   `ok` scale status. Setup improved at 16f from `63.17s` train plus `22.61s`
   heldout endpoint sequence build to `4.85s` train plus `1.90s` heldout by
   skipping benchmark-time full per-sample validation after exactness-unit-test
   coverage, vectorizing track-boundary coefficient reuse, and emitting packed
   delta-replace arrays directly from Gate4 affine candidate rows. A cut-array
   preallocation cleanup also verifies (`2.315/2.579/2.366/2.952 ms` total,
   `1.275x` total scale), but it is neutral on setup; a Python topology-reuse
   cache was negative at 16f and reverted. The previous keeper precomputed
   `owner -> boundary -> other owner` and verified with
   `2.127/3.592/2.368/2.916 ms` total, `1.371x` total scale, and `ok` status.
   The current keeper moves the single-slab direct row replay into the
   `world_foam_lane2_fused_slab_v0` C++ op and verifies with
   `2.258/2.154/2.464/2.966 ms` total, `1.314x` total scale,
   `1.364x` backward scale, and `1.040x` storage scale. 16f setup is now
   `3.48s` train plus `1.53s` heldout, down from the owner-membership
   `4.85s/1.90s`. Tensor-only native chunk merge and native first-owner
   selection were tested and reverted; the first regressed the 16f spot to
   `4.01s/2.12s` setup, and the second regressed the full 16f train setup to
   `4.22s` despite verifier `ok`. A native sorted-row packer improved 16f setup
   to `3.15s/1.28s` but changed endpoint segment counts (`222276` vs `222501`)
   and regressed warm timing, so it was also reverted. The high-cap
   sorted-vs-cut parity fixture now exists and passes in fallback plus
   extension-imported unittest modes (`Ran 7 tests ... OK`). A corrected native
   sorted-row op now passes that fixture and matches all 10 real 16f delta
   tensors against the cut-array native path, but remains disabled behind
   `GATE4_ENABLE_EXPERIMENTAL_NATIVE_SORTED_DELTA = False`: setup improves to
   `2.84-2.98s/1.15-1.17s`, yet repeated 16f spots regress warm timing to
   `8.248/7.847 ms` and `7.840/7.319 ms` total/backward. The restored default
   cut-array spot is `2.990/2.621 ms`. A same-process MPS probe proves all
   selected-device tensors equal, but keeping both cut and sorted tapes resident
   makes both paths slow (`7.913/7.545/6.769 ms` cut/sorted/cut median VJP), and
   naive `torch.mps.empty_cache()` before timing slows default cut-array to
   `8.010/7.554 ms` total/backward. Clean-process probe modes show lifetime
   order matters (`7.554 ms` cut-only when target/site MPS tensors are allocated
   before tape prep, `3.751 ms` after matching the trainer order), but sorted is
   still slow (`7.834 ms`; `gc`/device-clone negative; sync only moves both paths
   into a slower `4.5-4.8 ms` band). The explicit
   `--experimental-native-sorted-delta` full train/eval ladder also fails
   promotion despite identical PSNR/storage and better setup: robust verifier
   reports total mean/median scale `2.044x/2.124x`, backward mean/median scale
   `2.143x/2.284x`, and 16f warm `7.447/6.866 ms` total/backward versus the
   cut-array keeper's `2.966/2.640 ms`. A narrower native cut-prep fork removes
   Python cut-row assembly while still calling `gate4_delta_replace_from_cuts_cpu`,
   but it also fails robust promotion (`2.208x` total median scale,
   `2.208x` backward median scale; 16f `5.438/4.472 ms`). A same-path 16f
   device-layout screen kept PSNR fixed but confirmed packed framegroup16 is
   still the best existing representation (`4.097/3.558 ms` total/backward in
   that sweep; i16x3, owner-reduce, i16cols, i16x4, and framegroup64 were all
   slower). A minimal selected-device fork that leaves unused unpacked records,
   boundaries, and rays off MPS until final PSNR rendering also failed a 16f
   spot (`8.641/8.003 ms` vs same-session default `8.006/7.456 ms`, PSNR
   unchanged). I also wired the existing materialized i16x3 framegroup16 shader
   as an explicit mode; it passes the robust scale verifier (`1.424x` total
   mean scale, `1.418x` backward mean scale, storage `1.054x`, PSNR unchanged)
   but is too slow at 16f (`8.388/7.450 ms`) versus the clean native cut-array
   keeper (`2.966/2.640 ms`). A packed-materialized shader is better than
   i16x3 materialization and also passes the robust scale verifier
   (`0.909x/1.049x` total/backward mean scale, storage `1.040x`, PSNR
   unchanged), but it is still not a keeper (`5.757/5.209 ms` 16f
   total/backward; 16f spot `4.219/3.766 ms`). A high-site gradient-reduction
   fork also failed: reduce32 exceeded Metal's threadgroup memory cap
   (`34048 > 32768` bytes), and reduce24 launched but regressed 16f timing to
   `8.134/7.710 ms`; the reduction cap is back to `16`. A guarded
   smallrun16 packed shader that shrinks thread-private replay arrays to the
   observed Gate4 row cap also failed robust promotion: the wrapper/op exports
   and unit gate pass, but the ladder verifier reports `status=failed`, total
   mean/median scale `3.010x/3.012x`, backward mean/median scale
   `2.729x/3.070x`, and 16f `13.474/12.213 ms` despite unchanged PSNR/storage.
   A min-state recompute packed shader also failed the 16f spot screen: it
   removes stored `segment_trans`, `segment_alpha`, `weights`, and
   `segment_rgb`, then recomputes them in reverse, but the warmed spot is still
   `7.377/6.689 ms` versus the clean keeper's `2.966/2.640 ms` with unchanged
   PSNR. A packed scalar-launch VJP that keeps the packed int32 endpoint-row
   representation passes compile/export/unit gates and the full robust ladder
   (`status=ok`, total mean/median scale `1.363x/1.210x`, backward mean/median
   scale `1.515x/1.416x`, storage `1.040x`, PSNR unchanged), but it is not the
   new keeper: full-ladder medians are `3.097/6.107/2.216/3.746 ms` total and
   `2.133/3.939/1.892/3.020 ms` backward, the clean keeper remains
   `2.258/2.154/2.464/2.966 ms` total and `1.935/1.833/2.144/2.640 ms`
   backward, and 16f spot repeats vary from `2.679/2.291 ms` to
   `4.753/3.736 ms`. A CPU-rebase diagnostic flag that clones endpoint delta
   tensors into fresh contiguous CPU tensors before MPS transfer is also
   negative: native sorted + rebase is `6.605/6.138 ms` and native cut-prep +
   rebase is `6.226/5.456 ms` at 16f, with unchanged PSNR. A packed
   kernel-order selected-device diagnostic improved a slow same-window default
   (`4.326/3.866 ms` vs `7.890/5.768 ms` total/backward at 16f), but still does
   not beat the clean keeper (`2.966/2.640 ms`) and should stay diagnostic-only.
   A native C++ endpoint-record packer behind
   `--experimental-native-pack-records` passes build/unit/train-eval but is
   also non-keeper: full-ladder medians are `3.653/2.387/3.791/6.806 ms` total
   and `3.189/1.965/3.189/4.765 ms` backward, robust verification fails on a
   contaminated `4f` max/median outlier, and the same-window default 16f control
   is slow too (`6.143/5.249 ms`). Keep it as a diagnostic flag only; if packing
   is revisited, emit packed records inside the native row walk instead of
   adding a second pack pass. That native-emitted-pack follow-up now exists
   behind `--experimental-native-emitted-pack-records` and passes build/unit/full
   train-eval with unchanged PSNR, but it is also non-keeper: the raw scale
   verifier passes only because the 2f row is already slow, and the new
   reference-artifact verifier fails it against the clean native cut-array
   keeper (`16f` total/backward `4.837/3.987 ms`, `1.631x/1.510x` slower than
   keeper). `verify_framegroup16_timing_robust.py` now accepts
   `--reference-artifact` and default `1.20x` median non-regression limits, so
   future shader promotion must be both sublinear and not materially slower than
   the current keeper. A 16-thread launch fork for `frame_count<=16` was a hard
   negative (`32.245/24.774 ms` total/backward at 16f) and was reverted/rebuilt;
   the default 32-thread launch is restored. An opt-in
   `--defer-heldout-device` harness flag now exists to keep heldout MPS
   tape/targets out of the timed train loop, but its full reference-gated
   ladder is also negative (`19.120/12.181/9.518/7.396 ms` total medians and
   `7.396/6.057 ms` at 16f; `2.494x/2.294x` slower than keeper), so keep it as
   an allocation-order diagnostic only. A packed local-owner framegroup16 fork
   was implemented and passed scalar parity above the small-site reduction cap,
   but was removed from the hot variant after proving a hard performance
   negative: full-ladder medians were
   `8.060/6.926/5.751/69.411 ms` total and
   `5.847/5.576/4.407/63.366 ms` backward, with reference-verifier failures
   from `3.0x` slower at 2f to `24.0x` slower at 16f. This suggests the
   local-slot packed kernel shape itself is the problem; do not revive/promote
   it without a separate clean-room fork and reference-artifact proof.
   Timing artifacts now include `benchmark_environment` process snapshots and
   `verify_framegroup16_timing_robust.py --expected-frames` accepts single-frame
   spots for reference-artifact non-regression. Any artifact with
   `benchmark_environment.status=contended` is verifier-contaminated. This
   matters because post-cleanup default 16f controls stayed slow
   (`54.638/47.357 ms`, then `27.479/20.725 ms` total/backward medians versus
   the `2.966/2.640 ms` keeper), and the new metadata exposed active unrelated
   `ai_trader` Python jobs and clang work during the smoke.
   The later full `2/4/8/16f` control
   `2026-05-19_gate4_endpoint_record_default_control_envok_repeat20_render64_site24_2_4_8_16.json`
   is also rejected: it looks sublinear only because the low-frame rows are
   badly throttled (`68.615/60.824 ms` total/backward at 2f down to
   `8.132/6.865 ms` at 16f), and the metadata captured high-CPU `ai_trader`
   jobs at start/end. The environment classifier now splits blocking
   `blocking_processes` from low-CPU `background_processes`; background-only
   daemons are allowed, but high-CPU Python or Torch/Metal/MPS/PyTest jobs keep
   the artifact contaminated. The important exception is a periodic
   `run_btc15m_overnight_shadow_monitor.py` with `--toto-export-device mps` or
   `--toto-export-with-runtime-deps`: it is a blocker even while idle, because
   it can spawn a real MPS export during a timing window.
   Use the new fail-fast preflight before any promotion run:
   `train_eval_owner_run_tape.py --benchmark-environment-check-only` prints the
   snapshot and exits nonzero when blocked, while
   `--require-benchmark-environment-ok` aborts before tape construction if the
   machine is contended. The current check-only path exits `2` with multiple
   high-CPU `ai_trader` Python jobs and a PyTest process, so do not launch the
   next shader sweep yet. For the next clean window, use the wrapper
   `research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py`;
   it chains preflight, the default `2/4/8/16f` train/eval command, and the
   reference verifier. The blocked live invocation wrote
   `2026-05-19_framegroup16_promotion_preflight_blocked.promotion_summary.json`
   with `status=preflight_failed` and did not start train/eval.
   To let the gate wait for a clean machine, add
   `--wait-for-benchmark-environment-ok`; it now defaults to a one-hour timeout
   and 30-second polling. A short blocked wait smoke wrote
   `2026-05-19_framegroup16_promotion_wait_short_blocked.promotion_summary.json`
   with one recorded preflight attempt and no train/eval. Promotion summaries
   now also embed a compact `verify_result` when the verifier runs, including
   per-frame total/backward medians, contamination, and failure text, so a clean
   or failed promotion is auditable from one JSON file. The latest preflight is
   still blocked by a high-CPU PyTest plus a STAR UVT feature-kernel process, so
   wait before running the next timing gate. The wait preflight now writes its
   summary on every attempt; the live blocked smoke
   `2026-05-19_framegroup16_promotion_live_summary_blocked.promotion_summary.json`
   records `status=preflight_failed`, one attempt, and the top blocking process.
   Next fork: use full train/eval, not the
   standalone VJP probe, as the promotion gate while isolating sorted's MPS
   lifetime interaction, or change the endpoint representation so those rows
   are never materialized in Python. Do not cite the earlier truncating high-cap
   artifacts.
9. Keep updating `../agent_notes/key_learnings.md` only for surprising lessons;
   put ordinary progress and failed attempts in loose notes.

## TODO Map

- Browser trainer next gate: reorganize the all-pairs source-view backward into
  an image/tile-oriented pass before adding real windowed D-SSIM; then run a
  matched `converge47` versus Adam plus fixed-cap recycle/spawn ablation at the
  same splat count, samples, steps, and validation cadence. Do not promote the
  current optimizer foundation from startup/finite-update evidence alone.

- `trainer_landscape_unification.md`: trainer duplication, shared composition,
  validation media, and the mixed trainer/scheduler direction.
- `Clean_up_and_unify_interfaces.md`: older interface cleanup backlog. Check
  `trainer_landscape_unification.md` first before implementing broad trainer
  refactors.
- `video_token_implicit_camera_followups.md`: video-token implicit-camera
  followups, camera-swap work, render-size choices, and mixed multicam pretrain
  notes.
- `vjepa_f32_multicam_heldout_followups.md`: V-JEPA F32 multicam heldout
  diagnostics, overlap-aware splits, media panels, and best-heldout tracking.
- `reduce_render_memory_usage.md`: renderer memory work and feature-splatting
  memory levers.
- `alpha_mask_white_background_cheating.md`: alpha/background degeneracy in
  feature splatting.
- `radio_vipe_supervised_loss_todo.md`: supervised loss direction for RADIO/VIPE.
- `powerfoam_remaining_work_after_completion_audit_2026-05-06.md`: current
  PowerFoam post-audit backlog.
- `powerfoam_full_reproduction_todo.md`: historical PowerFoam completion audit;
  use the remaining-work file above for current work.
- `powerfoam_math_backlog_beyond_sweeps_2026-05-05.md` and
  `powerfoam_new_math_direction_backlog_2026-05-05.md`: PowerFoam research
  directions beyond simple schedule sweeps.
- `readme_and_positioning_followups.md`: README and positioning cleanup.

When closing a TODO, either edit the relevant file with the completion evidence
or add a new dated loose note that explains what changed and why the TODO is now
superseded.
