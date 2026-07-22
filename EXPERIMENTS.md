# DynaWorld Experiment Registry

This registry tracks active experiment lanes and the artifacts a new agent
needs to resume or interpret them. It is not a replacement for `BASELINES.md`:
benchmark claims still belong there.

## Status Labels

- **Active**: current lane; reasonable to continue.
- **Promoted**: current preferred row/config for that lane.
- **Blocked**: do not continue without fixing the named blocker.
- **Parked**: useful context, but not the next move by default.
- **Historical**: superseded; read only for archaeology.

## World Tubes / WorldFoam Paper Runner Ablation Lane

Status: **Promoted; shared gauge math retained, name proliferation stopped;
World Tubes breadth next, WorldFoam engineering parked**

Purpose: turn the World Tubes and WorldFoam paper ideas into reproducible
runner artifacts that can feed ablation tables, quality comparisons, and
failure-mode figures without hand-curated claims.

Current decision:

- Retain the Gauged UVT camera-ray bundle framework. Close only open-ended
  theory/name proliferation without a replayable failure. The invariant
  `UVT trace = pi_* Gamma^* world_primitive`, projective gauge domains,
  interval atlas, visibility strata, and compiled adjoint are the mathematical
  core of the World Tubes mainline.
- Treat WorldFoam as a distinct retained-depth optical-transfer challenger,
  not a competing default implementation. Reopen native WorldFoam work only
  after a broader heldout-quality win or a direct Metal optical-transfer parity
  gate.
- Do not add another local chart/gauge formalism unless a current orbit,
  visibility, or denominator certificate is falsified by a replayable case.
- Use `research_notes/renderer_lane_taxonomy.md` as the canonical distinction
  between Gauged UVT, World Tubes, STAR UVT, WorldFoam, and PowerFoam.

- Use report/verifier modules as the paper-runner contract: each runner writes
  `summary.json` and `summary.md`, exposes `summarize(...)`,
  `verify_...(...)`, `assert_...(...)`, and a `--verify-report` CLI, and has
  CPU-only tests that mutate stale/bad reports.
- The first green slice is the World Tubes decisive-demo fixture. It compares
  `per_frame_replay` with `compiled_interval_atlas` on a tiny two-trace
  projective cell atlas. It proves replay equivalence and interval compression
  in the report schema; it does not yet claim real-video quality or Metal
  throughput.
- The World Tubes decisive-demo report now also consumes the saved 128px/16f
  2048-tube visual-compare artifact as a real-video media row, copies the
  contact sheet and side-by-side video into the report artifact directory, and
  emits fallback/runtime/memory SVG artifacts. This fills the first
  real-video/media row for the table surface, but it is still a saved local
  visual row rather than a full benchmark sweep.
- The second green slice is the World Tubes visibility stress fixture. It
  records stable clean order, raw crossing collapse, stratified crossing repair,
  and forced fallback collapse in one verifier-backed artifact. This proves the
  stress-report shape and failure-boundary accounting, not real-video quality.
- The first WorldFoam paper-math slice is green. It implements the
  constant-density owner-run optical-transfer monoid, same-representation
  replay, analytic prefix/suffix VJP, finite-difference checks for
  beta/m/DeltaTau/sigma/color/run length, and the two-layer commutator swap
  probe. This is still a CPU fixture, not a Metal or real-video quality claim.
- The first shared table surface is green. It consumes the two World Tubes
  fixture reports, the WorldFoam optical-transfer fixture, the WorldFoam
  owner-run/Metal comparison report, the scoped paper-quality benchmark table,
  the matched three-seed `coffee_martini` heldout-camera table, and the 128px
  capacity visual comparison. The current 2026-07-11 report has nine green
  evidence rows, three ready representation rows, no missing IDs, and
  `paper_ready=true`.
- The first paper-quality benchmark table is a scoped
  `capacity_128_local_video_smoke`, not the full paper sweep. It derives
  matched media PSNR/L1 from side-by-side videos for all three representations
  and carries native metrics where the lane emitted them. This makes the
  ablation table reproducible, but the next scientific step is repeats,
  paper datasets, and heldout/novel-view splits.
- The WorldFoam owner-run/Metal comparison row is green, but the bridge scope
  is explicitly `contract_plus_visual_capacity_smoke`: it proves the CPU
  optical-transfer contract and the current Metal visual-capacity lane are both
  present, not that the Metal shader itself has full optical-transfer parity.
- Do not jump straight to native shader work from this fixture. The next
  runner should scale the paper-quality benchmark matrix first, then use
  measured bridge/native overhead to decide whether a new Metal entry point is
  justified.

Key files:

- `research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py`
- `tests/test_star_uvt_projective_decisive_demo_report.py`
- `research_experiments/star_uvt_feature_tubes/projective_visibility_stress_suite.py`
- `tests/test_star_uvt_projective_visibility_stress_suite.py`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/summary.json`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/summary.md`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/real_video_media/contact_sheet.jpg`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/real_video_media/side_by_side.mp4`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/real_video_media/fallback_heatmap.svg`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/real_video_media/runtime_bars.svg`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/real_video_media/memory_bars.svg`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_visibility_stress_suite/summary.json`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_visibility_stress_suite/summary.md`
- `research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py`
- `research_experiments/world_foam_lane2/test_cell_path_optical_transfer_fixture.py`
- `outputs/benchmarks/2026-07-08_worldfoam_cell_path_optical_transfer_summary.json`
- `research_experiments/world_foam_lane2/worldfoam_owner_run_metal_comparison_report.py`
- `research_experiments/world_foam_lane2/test_worldfoam_owner_run_metal_comparison_report.py`
- `outputs/benchmarks/2026-07-08_worldfoam_owner_run_metal_comparison_report/summary.json`
- `outputs/benchmarks/2026-07-08_worldfoam_owner_run_metal_comparison_report/summary.md`
- `research_experiments/paper_runner_suite/paper_quality_benchmark_table_report.py`
- `tests/test_paper_quality_benchmark_table_report.py`
- `outputs/benchmarks/2026-07-08_paper_quality_benchmark_table/summary.json`
- `outputs/benchmarks/2026-07-08_paper_quality_benchmark_table/summary.md`
- `research_experiments/paper_runner_suite/paper_runner_table_report.py`
- `tests/test_paper_runner_table_report.py`
- `outputs/benchmarks/2026-07-08_paper_runner_table_report/summary.json`
- `outputs/benchmarks/2026-07-08_paper_runner_table_report/summary.md`
- `research_experiments/paper_runner_suite/run_coffee_martini_matched_sweep.py`
- `research_experiments/paper_runner_suite/coffee_martini_matched_sweep_report.py`
- `tests/test_run_coffee_martini_matched_sweep.py`
- `tests/test_coffee_martini_matched_sweep_report.py`
- `outputs/benchmarks/2026-07-11_coffee_martini_matched_sweep/report/summary.json`
- `outputs/benchmarks/2026-07-11_coffee_martini_matched_sweep/report/summary.md`
- `outputs/benchmarks/2026-07-11_paper_runner_table_report/summary.json`
- `outputs/benchmarks/2026-07-11_paper_runner_table_report/summary.md`
- `research_experiments/paper_runner_suite/run_unified_paper_ablation.py`
- `src/train/paper_training_types.py`
- `src/train/paper_training_protocol.py`
- `tests/test_paper_training_protocol.py`
- `tests/test_unified_paper_ablation.py`
- `src/train_configs/paper_protocols/coffee_martini_full_300f_progressive_512_v1.jsonc`
- `src/train_configs/paper_protocols/coffee_martini_full_300f_fixed_512_pixel_matched_v1.jsonc`
- `src/train_configs/paper_protocols/coffee_martini_full_300f_progressive_global_shuffle_512_v1.jsonc`
- `outputs/benchmarks/2026-07-19_unified_paper_ablation_smoke_v2/coffee_martini_protocol_smoke_2step/seed_17/run_summary.json`
- `outputs/benchmarks/2026-07-19_unified_paper_ablation_smoke/coffee_martini_full_300f_smoke_1step/seed_17/run_summary.json`
- `research_notes/worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md`

Latest evidence:

- 2026-07-22 submission-spine update: evidence schema v1 now requires
  heldout LPIPS, sampled peak current/driver memory, serialized checkpoint
  bytes, synchronized compile/forward/backward/optimizer timing, and
  trace/event/fallback diagnostics for every lane. The fail-closed matrix
  runner emits JSON/CSV/Markdown/LaTeX/SVG artifacts. Live three-lane evidence
  smoke:
  `outputs/benchmarks/2026-07-22_unified_paper_evidence_smoke_v2/coffee_martini_protocol_smoke_2step/seed_17/run_summary.json`.
- The exact same-representation scaling artifact is verified at
  `outputs/benchmarks/2026-07-22_world_tubes_same_representation_scaling_f4_128_cap256/summary.json`.
  Across `F=4,8,16,32,64,128`, fixed payload growth is `1x` versus replay
  `32x`; final fixed/replay payload, compile, forward, and backward ratios are
  `0.03125`, `0.047677`, `0.181323`, and `0.392235`.
- The checked theorem-table artifact is
  `outputs/benchmarks/2026-07-22_world_tubes_theorem_table/summary.json`
  with generated Markdown/LaTeX. It deliberately labels scope as bounded
  camera-chart segments; full `360/720` multi-gauge transition remains
  unimplemented and is not claimed.
- The first progressive-512 seed-17 600-step comparison completed World Tubes
  and dynamic 3DGS. Heldout World Tubes is PSNR/SSIM/LPIPS
  `5.8945/0.03360/0.98461` with `124.58s` train wall and `3.114GB` peak MPS
  driver memory; dynamic 3DGS is `4.9110/0.28266/0.90229`, `142.58s`, and
  `20.557GB`. The WorldFoam lane is still required before this becomes a
  complete row; do not add it to `BASELINES.md` yet.
- The progressive-512 Coffee Martini rows for seeds `17/29/43` subsequently
  completed all three representations under
  `outputs/benchmarks/2026-07-22_world_tubes_submission_matrix_clean_v1/`.
  The next fixed-512 row was manually killed after severe unified-memory,
  compressor, swap, and `kernel_task` pressure destabilized the workstation;
  that partial row is invalid. No publication-scale local MPS row may be
  resumed without explicit approval. The unified runner now launches World
  Tubes, dynamic 3DGS, and WorldFoam in separate resumable child processes,
  validates metadata before merging lane reports, and requires authorization
  again inside each MPS child. This is containment work only: the conservative
  incident-calibrated high-risk guard remains unchanged until streaming or
  off-machine profiling supplies stronger evidence.
- CPU-only aggregation of the existing clean-source summaries accepted exactly
  the three progressive seeds and emitted 9 lane rows under
  `outputs/benchmarks/2026-07-22_world_tubes_submission_matrix_clean_v1/accepted_existing_evidence/`.
  `existing_evidence_summary.json` remains `partial_existing_evidence` and
  names the missing fixed seeds `17/29/43` plus global-shuffle seed `17`; no
  lane debris from the killed fixed run entered the table.

- Focused test:
  `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_decisive_demo_report.py -q`
  passed: `9 passed`.
- CLI smoke with saved real-video media and saved-artifact verifier:
  `PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py --fixture-only --include-saved-real-video --out-dir outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture`
  then
  `PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py --verify-report outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/summary.json`
  passed.
- Saved summary: `max_image_abs_error_vs_reference=0.0`,
  `min_psnr_vs_reference=120.0`,
  `compiled_to_replay_interval_entry_ratio=0.125`,
  `compiled_to_replay_memory_ratio=0.216`, `real_video_min_psnr=21.768529415130615`,
  `real_video_max_l1=0.054596319794654846`, and
  `real_video_min_artifact_count=5`.
- Visibility stress focused test:
  `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_visibility_stress_suite.py -q`
  passed: `7 passed`.
- Visibility stress CLI smoke and saved-artifact verifier:
  `PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_visibility_stress_suite.py --out-dir outputs/benchmarks/2026-07-08_star_uvt_projective_visibility_stress_suite`
  then
  `PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_visibility_stress_suite.py --verify-report outputs/benchmarks/2026-07-08_star_uvt_projective_visibility_stress_suite/summary.json`
  passed.
- Visibility stress saved summary: collapsed cases are
  `crossing_raw_interval` and `forced_fallback_ambiguous`; noncollapsed cases
  are `clean_orbit_ordered` and `crossing_stratified`; `max_quality_error` is
  `0.1867423951625824`, and `max_fallback_sample_fraction` is `1.0`.
- WorldFoam optical-transfer focused test:
  `PYTHONPATH=src/train uv run --with pytest python -m pytest research_experiments/world_foam_lane2/test_cell_path_optical_transfer_fixture.py -q`
  passed: `8 passed`.
- WorldFoam optical-transfer CLI smoke and saved-artifact verifier:
  `PYTHONPATH=src/train uv run python research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py --out outputs/benchmarks/2026-07-08_worldfoam_cell_path_optical_transfer_summary.json`
  then
  `PYTHONPATH=src/train uv run python research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py --verify-report outputs/benchmarks/2026-07-08_worldfoam_cell_path_optical_transfer_summary.json`
  passed.
- WorldFoam saved summary: checks are all `ok`; replay `render=0.0`,
  `element=0.0`, VJP `grad=2.4557592070983958e-11`, and commutator error
  `5.551115123125783e-17`.
- WorldFoam owner-run/Metal comparison focused test:
  `PYTHONPATH=src/train uv run --with pytest python -m pytest research_experiments/world_foam_lane2/test_worldfoam_owner_run_metal_comparison_report.py -q`
  passed: `6 passed`.
- WorldFoam owner-run/Metal comparison CLI smoke and saved-artifact verifier:
  `PYTHONPATH=src/train uv run python research_experiments/world_foam_lane2/worldfoam_owner_run_metal_comparison_report.py --out-dir outputs/benchmarks/2026-07-08_worldfoam_owner_run_metal_comparison_report`
  then
  `PYTHONPATH=src/train uv run python research_experiments/world_foam_lane2/worldfoam_owner_run_metal_comparison_report.py --verify-report outputs/benchmarks/2026-07-08_worldfoam_owner_run_metal_comparison_report/summary.json`
  passed.
- WorldFoam owner-run/Metal saved summary:
  `owner_run_metal_comparison_rows_ok=true`,
  `bridge_scope=contract_plus_visual_capacity_smoke`, and
  `paper_ready=false`.
- Paper quality benchmark table focused test:
  `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_paper_quality_benchmark_table_report.py -q`
  passed: `7 passed`.
- Paper quality benchmark CLI smoke and saved-artifact verifier:
  `PYTHONPATH=src/train uv run python research_experiments/paper_runner_suite/paper_quality_benchmark_table_report.py --out-dir outputs/benchmarks/2026-07-08_paper_quality_benchmark_table`
  then
  `PYTHONPATH=src/train uv run python research_experiments/paper_runner_suite/paper_quality_benchmark_table_report.py --verify-report outputs/benchmarks/2026-07-08_paper_quality_benchmark_table/summary.json`
  passed.
- Paper quality benchmark saved summary:
  `benchmark_scope=capacity_128_local_video_smoke`,
  `row_count=3`, `paper_ready=true`,
  `best_media_psnr_representation=world_tubes_star_uvt`, and
  `fastest_elapsed_representation=world_tubes_star_uvt`.
  Rows: World Tubes 2048 tubes / 60 steps / 17.077s /
  media PSNR 21.807 / media L1 0.0545; WorldFoam 2048 cells / 80 steps /
  32.811s / media PSNR 17.777 / media L1 0.0806; dynamic 3DGS 4096 Gaussians /
  60 steps / 89.572s / media PSNR 18.643 / media L1 0.0764.
- Paper table focused test:
  `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_paper_runner_table_report.py -q`
  passed: `7 passed`.
- Paper table CLI smoke and saved-artifact verifier:
  `PYTHONPATH=src/train uv run python research_experiments/paper_runner_suite/paper_runner_table_report.py --out-dir outputs/benchmarks/2026-07-08_paper_runner_table_report`
  then
  `PYTHONPATH=src/train uv run python research_experiments/paper_runner_suite/paper_runner_table_report.py --verify-report outputs/benchmarks/2026-07-08_paper_runner_table_report/summary.json`
  passed.
- Paper table saved summary after the quality-table row: eight green evidence
  rows, three representation rows, `paper_ready=true`, and no missing IDs.

Next runner TODOs:

- Unified paper-ablation software status: green at the pre-incident smoke
  scale only. The 4-frame/two-stage MPS
  smoke completed World Tubes, dynamic 3DGS, and WorldFoam with exact shared
  cost `4 frames / 30,720 pixels`; optimizer-update times were `0.298s`,
  `0.299s`, and `0.608s`, respectively. WorldFoam also completed its
  optimizer-state-preserving `128 -> 256` cell transition. The all-300-frame
  one-step smoke also completed all three lanes, full train/heldout evaluation,
  media, and offline W&B. These timings are mechanical smokes, not benchmark
  rankings.
- Run `coffee_martini_full_300f_progressive_512_v1` and the exact
  target-pixel-matched fixed control. Then run the global-shuffle sampler
  ablation, seeds 17/29/43, additional camera triplets, and Neural3D scene
  breadth. Keep `fast_exploration` as the throughput row and deterministic
  policies as separately labeled correctness audits.
- Do not claim native 2704x2028 support from the eager 512-wide runner. Native
  promotion requires streamed K-frame targets/rays and streamed evaluation;
  the dependency chain is in `TODO/unified_paper_ablation_pipeline.md`.
- The first real-dataset protocol smoke is green on Neural3D
  `coffee_martini`: train `cam04`/`cam09`, hold out `cam06`, and use
  `neural_3d_llff_relative_pinhole` calibration. The saved protocol report is
  `outputs/benchmarks/2026-07-11_coffee_martini_train2_holdout1_protocol/summary.json`.
  The matched three-seed table is now complete at 128px/16f/40 steps/1024
  primitives with offline W&B media and separate train/heldout PSNR/SSIM/L1.
  Saved report:
  `outputs/benchmarks/2026-07-11_coffee_martini_matched_sweep/report/summary.json`.
  Mean heldout PSNR: World Tubes `6.3863 +/- 0.0154`, paper-clean WorldFoam
  `5.6311 +/- 0.0000`, dynamic 3DGS `4.9544 +/- 0.0004`. World Tubes used the
  promotable deterministic-quality policy and wins the scoped table. All
  verifier gates pass, including exact seeds 17/29/43, camera split, matched
  budget, clean WorldFoam initialization, media, and W&B backing. This is a
  complete one-scene/one-split paper table, not a multi-scene SOTA claim.
- Extend the same protocol to more `coffee_martini` camera triplets and then
  the remaining Neural3D scenes before making a broad quality claim. Keep
  heldout-camera PSNR primary and report deterministic-quality correctness
  timing separately from the direct-atomic throughput kernel.
- Decide whether WorldFoam needs a native optical-transfer Metal entry point
  after the scaled runner shows whether bridge/native overhead or quality is
  the limiting factor.

## STAR UVT Feature-Tube Support And Binfix Lane

Status: **Active / binner repaired; prefix-alpha train measured; broadened ownership next**

Purpose: make STAR UVT support-changing feature tubes actually cover selected
target regions before spending engineering effort on STAR Softmax-GS ports or a
WorldFoam promotion.

Current decision:

- The selected-patch diagnostic found a renderer bug: chunk-shifted moving
  support tubes had analytic target alpha but were culled by sparse binning.
  `tube_bounds` now accepts valid small determinants with
  `max(eps^2, 1e-20)` and falls back to `abs(m)+domain` bounds for shifted
  local chunks.
- Keep STAR support as the near-term mainline. Softmax-GS remains a dynamic-GS
  probe; WorldFoam remains a challenger pending a matched tournament row.
- Dense/media transfer is now measured: the binfix improves whole-frame
  support but does not close the visibility/composition gap. The compact
  visibility-prefix tape is now measured too: selected born support is present
  and usually dominant on selected target rays. The first prefix-alpha train
  moves selected-ray contribution but lands essentially on the same dense
  plateau, so the next STAR move should broaden ownership/coverage sampling or
  change the support distribution, not simply turn the same local alpha knob
  harder.

Key files:

- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`
- `tests/test_star_uvt_feature_binning.py`
- `research_experiments/star_uvt_feature_tubes/support_target_patch_diagnostic.py`
- `research_experiments/star_uvt_feature_tubes/visibility_prefix_tape_diagnostic.py`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetarea2_binfix_from1500_lr001_50step_media.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetarea2_binfix_prefixalpha_from1500_lr001_50step_media.jsonc`
- `agent_notes/loose_notes/2026-05-26_18-52-25_star_uvt_binner_binfix_train.md`

Latest evidence:

- Focused binner regression:
  `PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_feature_binning.py -q`
  passed.
- Repaired three-case selected-patch diagnostic:
  `outputs/benchmarks/2026-05-26_star_uvt_support_target_patch_diagnostic_binfix.md`.
  Targetinit/targetalpha/targetarea2 normal patch PSNR is
  `4.606/4.686/4.684`, forced-alpha PSNR is `14.529/14.694/14.677`, and
  selected-only alpha is about `0.30` instead of the previous `0.0`.
- First repaired targetarea2 50-step train:
  `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_binfix_50step_media.json`.
  It passes with zero overflow, max tile count `110/128`, loss
  `0.889263 -> 0.863064`, support-target-area loss
  `0.253626 -> 0.217254`, feature loss `0.612217 -> 0.610967`, and RGB-probe
  PSNR `24.253 -> 24.453`.
- Post-train selected-patch diagnostic:
  `outputs/benchmarks/2026-05-26_star_uvt_support_target_patch_diagnostic_targetarea2_binfix_train.md`.
  Selected patches reach normal/forced/oracle PSNR `6.644/19.452/26.994`,
  patch alpha mean `0.481`, and selected-only alpha `0.444`.
- Post-train dense support diagnostic:
  `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_targetarea2_binfix_dense_support.md`.
  Dense normal/forced/oracle PSNR is `7.269/14.736/21.439`, alpha mean is
  `0.3456`, alpha `>0.1` covers `75.4%` of pixels, and the best raw
  logit-opacity bias reaches only `8.039` PSNR. Versus the pre-binfix
  targetarea2 repair row (`6.507/14.085/21.627`, alpha `>0.1` `65.7%`), the
  binner fix is a real coverage gain, but the remaining forced-alpha/oracle gap
  still points at visibility/prefix/composition rather than simple opacity
  pressure.
- Visibility-prefix tape diagnostic:
  `outputs/benchmarks/2026-05-26_star_uvt_targetarea2_binfix_visibility_prefix_tape.md`.
  On 256 selected support-target rays, normal/forced/oracle PSNR is
  `6.522/19.129/26.831`; final alpha mean is `0.4755`; selected born tubes have
  weight share `0.9308`, are absent on `0.0%` of rays, prefix-hidden on only
  `1.6%`, and are the top contributor on `95.7%`. Selected support is not
  primarily hidden by older tubes; it owns these rays but does not yet produce
  enough final alpha/black-background contribution.
- Prefix-alpha compositing train:
  `outputs/benchmarks/2026-05-28_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_binfix_prefixalpha085w2_50step_media.json`.
  It passes with zero overflow, loss `1.285825 -> 1.210325`,
  support-target-area loss `0.253626 -> 0.219786`, prefix-alpha loss
  `0.198281 -> 0.172906`, selected weight mean `0.4114 -> 0.4419`, and final
  alpha mean `0.4456 -> 0.4751` on the prefix-loss sample. The matching dense
  diagnostic
  `outputs/benchmarks/2026-05-28_star_uvt_birthsplit_targetarea2_binfix_prefixalpha085w2_50step_dense_support.md`
  is effectively flat against the no-prefix binfix row:
  normal/forced/oracle PSNR `7.262/14.732/21.438`, alpha `>0.1` `75.4%`, best
  raw-opacity-bias PSNR `8.037`. The matching prefix tape
  `outputs/benchmarks/2026-05-28_star_uvt_targetarea2_binfix_prefixalpha085w2_50step_visibility_prefix_tape.md`
  shows local selected weight share `0.9381`, top selected `96.9%`, and
  selected-weight mean `0.4374`. Conclusion: prefix-alpha is a useful
  measurement/control surface, but not enough by itself to close dense RGB.

## Dynamic GS Softmax-GS Renderer Probe

Status: **Active / shader proven; bounded-tape scalar backward; tiny heldout row positive; repeat/scale mixed**

Purpose: test whether Softmax-GS overlap-aware compositing improves dynamic
Gaussian splat quality, temporal stability, or primitive efficiency before
spending STAR/WorldFoam effort on the same idea.

Current decisions:

- Keep the probe on RGB/F3 dynamic GS first. F32 feature splats and STAR UVT
  ports wait until the dynamic-GS result is positive.
- Use `fast_mac.depth_mode="center_camera_z"` for Softmax-GS. The old
  rank-depth signal remains the default for vanilla/no-op paths.
- Treat `softmax_gs_enabled=true` training as a hybrid native route: native
  recompute remains available, and bounded tape can drive color plus selected
  geometry/opacity/depth gradients when `softmax_gs_tape_k > 0`. Full-tape
  coverage is exact; bounded K is an explicit approximation.
- The bounded top-K tape now exists in the Torch reference and in the
  `v5_softmax_gs` Metal ABI/kernel path for fast and overflow tiles. Backward
  consumes it for color and selected scalar gradients. K=8 is too lossy in the
  first 50-step diagnostic; K=16 is the current small-row winner; K=32 trains
  but does not improve the 50-step source-view result.
- The first matched multicam RGB-pyramid heldout diagnostic is positive for
  K=16 on a tiny row: final train loss is tied with the no-op control, while
  heldout PSNR jumps `4.7369 -> 11.7255`. This is still too small to promote,
  but it is now stronger than source-view-only evidence.
- Repeat/scale does not cleanly confirm the tiny heldout win. The 64px/4f/512
  row loses heldout PSNR, and the practical 128px/16f/512 stride16 row only
  nudges heldout PSNR while losing heldout SSIM and train-view metrics.

Key files:

- `research_notes/gaussian_splatting_papers/2026-05-25_short_term_softmax_gs_plan.md`
- `research_notes/gaussian_splatting_papers/2026-05-25_long_term_splats_vs_worldfoam_plan.md`
- `research_notes/gaussian_splatting_papers/2604_27437_softmax_gs_dynaworld_integration.md`
- `research_experiments/softmax_gs/reference.py`
- `src/train/renderers/fast_mac.py`
- `third_party/fast-mac-gsplat/variants/v5_softmax_gs/`
- `tests/test_softmax_gs_reference.py`
- `tests/test_softmax_gs_metal_forward.py`
- `src/train_configs/local_mac_softmax_gs_enabled_smoke_32_2f_64splats.jsonc`
- `src/train_configs/local_mac_softmax_gs_noop_smoke_32_2f_64splats.jsonc`
- `src/train_configs/local_mac_softmax_gs_noop_smoke_64_4f_128splats.jsonc`
- `src/train_configs/local_mac_softmax_gs_noop_diagnostic_64_4f_128splats_10step.jsonc`
- `src/train_configs/local_mac_softmax_gs_enabled_diagnostic_64_4f_128splats_10step.jsonc`
- `src/train_configs/local_mac_softmax_gs_enabled_overflow_smoke_32_2f_64splats_2step.jsonc`
- `src/train_configs/local_mac_softmax_gs_enabled_tapecolor_diagnostic_32_2f_64splats_5step.jsonc`
- `src/train_configs/local_mac_softmax_gs_enabled_tapescalar_diagnostic_seed17_64_4f_128splats_50step.jsonc`
- `src/train_configs/local_mac_softmax_gs_enabled_tapescalar_k16_diagnostic_seed17_64_4f_128splats_50step.jsonc`
- `src/train_configs/local_mac_softmax_gs_enabled_tapescalar_k32_diagnostic_seed17_64_4f_128splats_50step.jsonc`
- `src/train_configs/local_mac_softmax_gs_noop_diagnostic_seed17_64_4f_128splats_50step.jsonc`
- `src/train_configs/local_mac_softmax_gs_enabled_diagnostic_seed17_64_4f_128splats_50step.jsonc`
- `src/train_configs/local_mac_multicam_softmax_gs_noop_rgb_pyramid_64_4f_128splats_20step.jsonc`
- `src/train_configs/local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_64_4f_128splats_20step.jsonc`

Latest evidence:

- Focused tests pass:
  `28 passed` for fast-mac depth signal, CPU Softmax-GS reference, MPS
  Softmax-GS forward/native fast+overflow backward/bounded tape/full-tape
  color+scalar backward, and feature background gates.
- No-op `v5_softmax_gs` matches vanilla `v5` on MPS with forward max error
  `0.0` and gradient max error `2.98e-8`.
- Forward-only Softmax-GS fixes the synthetic same-depth two-splat swapped
  color artifact on MPS: vanilla swap max diff `4.7309e-1`, Softmax-GS swap
  max diff `2.3842e-7`.
- Historical scaffold evidence: the old Torch recompute training route matched
  Metal forward to `2.98e-7`, produced finite gradients, and caught the
  rationalized-rescale numerical fix. It has now been superseded by native
  recompute backward.
- The CPU reference now has the first executable native-backward contract:
  `softmax_gs_contribution_tape(...)` reconstructs `weights @ features`,
  preserves final alpha, and gives exact color gradients. Focused reference
  tests pass `11 passed`. The reference also has
  `softmax_gs_bounded_contribution_tape(...)`, which selects exact top-K final
  contribution weights in ray order and returns residual mass that bounds
  unit-feature output error. Scalar reverse propagation through
  absorbance/exponent/depth is covered by the native recompute bridge and, for
  selected rows, by the bounded-tape scalar backward. The bounded-tape shader
  ABI now exists and the backward path consumes it for color and scalar
  gradients.
- The first native shader-side backward cut now exists for fast and overflow
  tiles. It recomputes per-pixel Softmax-GS scalar state in Metal and matches
  the Torch recompute reference on tiny MPS projected scenes, including
  means/conics/colors/opacities/depths with nonzero `gamma`. The forced
  overflow test sets `max_fast_pairs=1` and confirms the enabled overflow
  backward route. Focused gate: `28 passed` across depth-signal, Softmax-GS
  reference, Softmax-GS MPS, and feature-background tests.
- The bounded top-K tape has a first Metal lowering:
  - API:
    `rasterize_softmax_gs_bounded_tape(...) -> selected_ids, selected_weights, residual_weight, final_alpha`.
  - Fast-tile and forced-overflow MPS tests match the Torch reference top-K
    IDs/weights, residual mass, and final alpha.
  - Full-tape color+scalar gradient tests match the Torch reference in both
    fast and forced-overflow modes. This proves the selected-row bounded-tape
    backward consumer. It is exact when `softmax_gs_tape_k` covers every active
    contributor and approximate otherwise.
  - Build command:
    `( cd third_party/fast-mac-gsplat/variants/v5_softmax_gs && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )`.
  This proves the tape ABI/kernel lowering and the full-tape backward
  replacement. Bounded K still needs quality/residual characterization.
- Native enabled trainer smokes now complete:
  - one-step enabled `32px/2f/64 splats`: initial loss `0.4373`, step-1 loss
    `0.4270`, `8.59s/it` including first-use compile.
  - five-step enabled diagnostic: initial loss `0.4370`, step losses
    `0.4224, 0.4169, 0.4362, 0.4482, 0.4445`, tqdm mean `2.10it/s`.
  - same-session five-step no-op diagnostic: initial loss `0.4381`, final loss
    `0.4324`, tqdm mean `1.62it/s`.
  - forced-overflow two-step enabled smoke
    `local_mac_softmax_gs_enabled_overflow_smoke_32_2f_64splats_2step.jsonc`:
    initial loss `0.4374`, step losses `0.4775, 0.4486`, W&B disabled,
    `max_fast_pairs=1`.
  - post-bounded-tape ABI five-step enabled diagnostic
    `local_mac_softmax_gs_enabled_diagnostic_32_2f_64splats_5step.jsonc`:
    initial loss `0.4382`, final loss `0.4165`, W&B disabled.
  - post-bounded-tape ABI forced-overflow two-step enabled smoke:
    initial loss `0.4382`, step losses `0.4394, 0.4529`, W&B disabled.
  - tape-scalar diagnostic
    `local_mac_softmax_gs_enabled_tapecolor_diagnostic_32_2f_64splats_5step.jsonc`:
    `softmax_gs_tape_k=8`, initial loss `0.4382`, final loss `0.4190`, W&B
    disabled.
  These are local mechanical smokes with W&B disabled/offline, not benchmark or
  quality rows.
- A matched 64px/4f/128-splat 10-step offline W&B diagnostic now exists with
  media:
  - no-op control config:
    `local_mac_softmax_gs_noop_diagnostic_64_4f_128splats_10step.jsonc`;
    initial loss `0.4330`, step losses
    `0.4226, 0.4157, 0.4461, 0.4456, 0.4115, 0.4449, 0.4207, 0.4079, 0.4423, 0.4177`,
    tqdm mean `2.80s/it`, offline run
    `wandb/offline-run-20260525_193712-1lra1t7t`.
  - enabled native recompute config:
    `local_mac_softmax_gs_enabled_diagnostic_64_4f_128splats_10step.jsonc`;
    initial loss `0.4339`, step losses
    `0.4339, 0.4455, 0.4373, 0.4414, 0.4322, 0.4457, 0.4449, 0.4308, 0.4578, 0.4413`,
    tqdm mean `1.27s/it`, offline run
    `wandb/offline-run-20260525_193830-fu0df3ks`.
  W&B was run offline because `WANDB_API_KEY` was unset locally. Treat this as
  a source-view/media diagnostic and speed sanity check, not a quality
  promotion.
- Fresh post-overflow-shader matched 64px/4f/128-splat 10-step offline W&B
  diagnostics:
  - no-op control: initial loss `0.4337`, final loss `0.4456`, tqdm mean
    `2.43s/it`, offline run `wandb/offline-run-20260525_195019-27rj83gw`.
  - enabled recompute: initial loss `0.4342`, final loss `0.4198`, tqdm mean
    `1.60s/it`, offline run `wandb/offline-run-20260525_195115-tn9t3nby`.
  This fresh tiny run is directionally nicer for Softmax-GS than the previous
  one, but it remains a source-view diagnostic with too few steps to promote.
- TokenGS now normalizes `train.seed` (default `17`) and calls
  `torch.manual_seed(...)` at trainer startup, so matched renderer diagnostics
  can share model init and temporal sampling. Seeded 50-step 64px/4f/128-splat
  offline W&B diagnostics:
  - no-op control config:
    `local_mac_softmax_gs_noop_diagnostic_seed17_64_4f_128splats_50step.jsonc`;
    initial loss `0.4338`, final loss `0.1467`, tqdm mean `1.65it/s`,
    offline run `wandb/offline-run-20260525_200015-s04n74di`.
  - enabled recompute config:
    `local_mac_softmax_gs_enabled_diagnostic_seed17_64_4f_128splats_50step.jsonc`;
    initial loss `0.4338`, final loss `0.1512`, tqdm mean `1.32it/s`,
    offline run `wandb/offline-run-20260525_200101-xd4sm546`.
  This cleaner source-view diagnostic is neutral to slightly negative for
  enabled Softmax-GS at this tiny scale. Do not promote or port to STAR from
  this result.
- Post-selected-scalar tape 50-step diagnostics:
  - K=8 selected scalar tape config:
    `local_mac_softmax_gs_enabled_tapescalar_diagnostic_seed17_64_4f_128splats_50step.jsonc`;
    `softmax_gs_tape_k=8`, initial loss `0.4338`, final loss `0.2026`, tqdm
    mean `1.38it/s`, offline run
    `wandb/offline-run-20260525_204628-sk2fc3ne`.
  - K=16 selected scalar tape config:
    `local_mac_softmax_gs_enabled_tapescalar_k16_diagnostic_seed17_64_4f_128splats_50step.jsonc`;
    run with `GSP_TAPE_CAP=16`, initial loss `0.4338`, final loss `0.1472`,
    tqdm mean `3.19it/s`, offline run
    `wandb/offline-run-20260525_204816-oip27eka`.
  - K=32 selected scalar tape config:
    `local_mac_softmax_gs_enabled_tapescalar_k32_diagnostic_seed17_64_4f_128splats_50step.jsonc`;
    run with `GSP_TAPE_CAP=32`, initial loss `0.4338`, final loss `0.1588`,
    tqdm mean `3.63it/s`, offline run
    `wandb/offline-run-20260525_205435-wy8r4v9l`.
  K=8 is too lossy for the tiny 50-step source-view diagnostic. K=16 recovers
  the earlier seeded no-op/recompute bracket (`0.1467` no-op, `0.1512`
  enabled recompute) while using the selected scalar tape path. K=32 does not
  improve the endpoint on this row and is not the current default. This is
  still source-view-only and not a quality promotion.
- First matched multicam RGB-pyramid heldout diagnostic, 64px/4f/128 splats,
  20 steps, seed 17, W&B offline:
  - no-op control config:
    `local_mac_multicam_softmax_gs_noop_rgb_pyramid_64_4f_128splats_20step.jsonc`;
    initial loss `0.5910`, final train loss `0.2261`, offline run
    `wandb/offline-run-20260525_210925-39a0kpp2`; eval PSNR/SSIM:
    train view0 `13.4197/0.1148`, train view1 `14.3734/0.1679`,
    heldout camera_0040 `4.7369/0.0503`.
  - enabled K=16 selected scalar tape config:
    `local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_64_4f_128splats_20step.jsonc`;
    run with `GSP_TAPE_CAP=16`; initial loss `0.5910`, final train loss
    `0.2262`, offline run `wandb/offline-run-20260525_211008-vfwslw6q`;
    eval PSNR/SSIM: train view0 `13.4502/0.0944`, train view1
    `12.3880/0.1191`, heldout camera_0040 `11.7255/0.0794`.
  - Timing is noisy but same envelope at this scale: no-op step-20
    total/backward/raster `291/86/58ms`; enabled K=16 `372/97/48ms`.
    The heldout gain is promising, but the row is tiny, RGB-pyramid
    conditioned, and only 20 steps; do not update `BASELINES.md` or port to
    STAR/WorldFoam from this alone.
- Repeat/scale diagnostic for the same split exposed two facts. First, the
  original wider scale attempts hit an MPS `MPSNDArrayDescriptor
  sliceDimension` assertion before rasterization. This is now localized to
  large-memory `nn.MultiheadAttention`: a synthetic MPS MHA repro is safe at
  32,768 memory tokens and crashes at 40,960. `QueryCrossAttentionBlock` now
  uses a manual batch-first MHA fallback on MPS above 32,768 memory tokens;
  `tests/test_mps_safe_cross_attention.py` covers CPU parity and an MPS
  40,960-token smoke. A 128px/16f forward/tape smoke now completes. The
  unstrided 128px/16f training row was interrupted after 3/20 steps at 9:47,
  so the practical local scale pair uses `video_feature_token_stride=16`.
  Second, the primitive-count repeat at 64px/4f/512 splats is not a
  heldout-PSNR repeat of the tiny win:
  - no-op control config:
    `local_mac_multicam_softmax_gs_noop_rgb_pyramid_64_4f_512splats_20step.jsonc`;
    initial loss `0.5817`, final train loss `0.2511`, offline run
    `wandb/offline-run-20260525_212845-8rj3swm6`; eval PSNR/SSIM:
    train view0 `11.8441/0.1112`, train view1 `12.0649/0.1218`,
    heldout camera_0040 `12.5002/0.0817`; step-20 total/backward/raster
    `707/155/97ms`.
  - enabled K=16 selected scalar tape config:
    `local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_64_4f_512splats_20step.jsonc`;
    run with `GSP_TAPE_CAP=16`; initial loss `0.5818`, final train loss
    `0.2378`, offline run `wandb/offline-run-20260525_212923-wbr8y46t`;
    eval PSNR/SSIM: train view0 `12.8191/0.0917`, train view1
    `12.0651/0.1221`, heldout camera_0040 `11.8847/0.0950`; step-20
    total/backward/raster `554/140/102ms`.
  Enabled K=16 improves source/train loss at larger primitive count and
  slightly improves heldout SSIM, but it loses heldout PSNR to no-op by
  `0.6155dB`. Treat the first 128-splat heldout jump as not yet repeated.
- Practical 128px/16f/512 stride16 repeat:
  - no-op control config:
    `local_mac_multicam_softmax_gs_noop_rgb_pyramid_128_16f_512splats_stride16_20step.jsonc`;
    initial loss `0.5843`, final train loss `0.2577`, offline run
    `wandb/offline-run-20260525_220100-zod704i9`; eval PSNR/SSIM:
    train view0 `10.9996/0.1416`, train view1 `12.2710/0.1729`,
    heldout camera_0040 `12.1234/0.1244`; step-20 total/backward/raster
    `1865/336/122ms`.
  - enabled K=16 selected scalar tape config:
    `local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_128_16f_512splats_stride16_20step.jsonc`;
    run with `GSP_TAPE_CAP=16`; initial loss `0.5843`, final train loss
    `0.2504`, offline run `wandb/offline-run-20260525_220309-pkrvtzda`;
    eval PSNR/SSIM: train view0 `10.8973/0.1372`, train view1
    `11.6462/0.1581`, heldout camera_0040 `12.2092/0.1088`; step-20
    total/backward/raster `1107/197/65ms`.
  Enabled K=16 gets a tiny heldout-PSNR nudge (`+0.0858dB`) and a slightly
  better final train loss, but loses heldout SSIM and both train-view metrics.
  This is mixed evidence, not a promotion.
- Bounded-tape residual coverage diagnostic now exists:
  `research_experiments/softmax_gs/diagnose_tape_coverage.py`, with artifact
  `outputs/benchmarks/2026-05-25_softmax_gs_tape_coverage_64_4f_512_k16/`.
  Command:
  `PYTHONPATH=src/train GSP_TAPE_CAP=16 .venv/bin/python research_experiments/softmax_gs/diagnose_tape_coverage.py src/train_configs/local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_64_4f_512splats_20step.jsonc --train-steps 20 --k-values 1,2,4,8,16 --views train0,train1,heldout0 --output-dir outputs/benchmarks/2026-05-25_softmax_gs_tape_coverage_64_4f_512_k16`.
  The 20-step diagnostic says K=16 covers almost all selected mass after this
  tiny train: residual/alpha mean/p99 is train0 `0.000652/0.008290`, train1
  `0.000879/0.009899`, heldout `0.001930/0.012332`. K=8 is already small on
  train views but leaves a larger heldout tail (`0.040167` mean,
  `0.112505` p99). Therefore the 512-splat heldout-PSNR miss is unlikely to be
  explained only by K=16 tail truncation; it looks more like Softmax
  compositing/optimization is changing source fit in a way that does not
  transfer to heldout PSNR on this split.
- Historical Torch-fallback smokes remain useful only as pre-native
  chronology; do not cite them as the current route or as quality evidence.

Next useful experiment:

- Do not port Softmax-GS into STAR or WorldFoam from the current evidence.
  Short-term Softmax work should either run a stronger dynamic-GS heldout gate
  with learned Softmax-GS parameters or stop and return effort to STAR support
  / WorldFoam challenger gates. The MPS forward blocker is fixed; the remaining
  blocker is quality evidence. If heldout PSNR/SSIM does not repeat under a
  stronger gate, leave Softmax-GS as an opt-in dynamic-GS renderer option.

## STAR UVT Source-View Overfit

Status: **Active / promoted direct-atomic path**

Purpose: prove the time-tubed STAR UVT formulation can fit long clips without
the old per-frame dynamic-splat scaling pain.

Current decisions:

- Use `sample_emission_mode=direct_atomic`, `reduction_mode=index_add` as the
  practical path.
- Do not promote deterministic compact backward until its load-growth/backward
  row is competitive.
- Treat source-view overfit as source-view evidence, not novel-view proof.

Key configs:

- `src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_directatomic_200step.jsonc`
- `src/train_configs/star_uvt_highmotion_hlaZbH_64f_512_directatomic_multires256c200_50fine.jsonc`
- `src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_tilepair_suffix_keyseg_20step_profile.jsonc`

Key W&B / baseline rows:

- `jba7kztn`: 64f/256px/32768 tubes, direct atomic, PSNR `29.823`.
- `4r2x8s3c`: 64f/512px multires, direct atomic, PSNR `29.138`.
- `641gxm9l`: deterministic compact probe, not promoted.
- `BASELINES.md` Tier 1 STAR UVT rows.

Key results/logs:

- `star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_32768t_cap256_directatomic_200step.json`
- `star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c200_50fine.json`
- `star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_256_64f_8192t_cap256_tilepair_suffix_keyseg_20step.json`
- `agent_notes/loose_notes/2026-05-17_17-01-49_star_uvt_thread_closeout.md`
- `agent_notes/loose_notes/2026-05-17_23-19-05_shader_audit_and_fast_overfit_plan.md`

Next useful experiment:

- Bridge STAR UVT from first-class overfit into manifest/dataset training.
- Run direct-atomic old-formulation frame-count sweeps separately to isolate
  scratch-memory savings from STAR time-tubed scaling.

## STAR UVT Feature Tubes

Status: **Active / Gate 4 quality bracket failing; STAR V-JEPA target-grid sparse-forward diagnostic plus RGB oracle now passing**

Purpose: port the fast F32 feature-splat lessons into the STAR UVT time-tube
representation so F32/F64 tube features can render through `FeatureToColor`
without falling back to the expensive projected feature-raster path.

Current decisions:

- Keep this feature path separate from RGB `star_uvt_v0`; that renderer is
  hardcoded around `float3` color and RGB gradient reducers.
- Direct feature path first. Deterministic compact feature backward and
  fixedbin promotion come after a trainable direct feature kernel exists.
- Treat render/loss microbatching as part of the contract, because the dense
  `[T,H,W,F]` feature image and `FeatureToColor` graph can dominate memory.
- The camera-ray bundle framing now has a concrete gauge-invariance artifact:
  `outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.md`.
  It integrates the same orbit-camera spacetime Gaussian in ordinary depth and
  log-depth fiber gauges. With the correct Jacobian, max relative error is
  `3.50e-13`; without that Jacobian, the relative error is at least `0.600`.
  This protects the key math `UVT trace = pi_* Gamma^* world_primitive` from
  becoming only prose.
- The clean-derivatives side now has a matching gauge-gradient artifact:
  `outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.md`.
  Primitive gradients for mean, log-precision, and log-amplitude match across
  ordinary-depth/log-depth gauges to `2.33e-12` relative error with the
  Jacobian; the missing-Jacobian gradient control is wrong by at least `0.592`.
- The shared-work / bandwidth goal now has an aggregate audit:
  `outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.md`.
  It verifies the saved orbit and trained high-motion reports, then locks
  direct active-goal ratios: orbit fixed payload grows `1.0x` while per-frame
  replay grows `8.0x` for an explicit payload-growth ratio of `0.125`;
  trained shared interval entries grow at most `1.462x` while per-frame replay
  entries grow at least `9.852x`, giving a shared/replay interval-entry growth
  ratio of `0.148`; final trained shared/per-frame trace-count, forward, and
  backward ratios are `0.1`, `0.266`, and `0.094`. The restored default orbit
  artifact now verifies `8/16/32/64` frames with final fixed/per-frame payload,
  trace, and segment ratios all `0.0625`, forward ratio `0.117`, and backward
  ratio `0.158`.
  The audit contract also requires exposure/rolling forward, exposure/rolling
  backward, and differentiable mixed-fallback backward artifacts; the saved
  aggregate artifact verifies by CLI.
- The active goal now has an explicit progress audit:
  `outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.md`.
  It now verifies gauge invariance, gauge gradients, one- and two-parameter
  camera-family gauges, one- and two-parameter camera-family shared-work
  scaling, Q2 Metal lowering/chain-rule/materialized-batch/native-eval/native
  interval forward/backward, Q2 tile/order and active-set strata, checked-in
  real-video active-set distribution, trainer smokes, multiscene real-video
  matrices, quality/media tethers, the real-video acceptance envelope, the
  real-video timing-variance envelope, the real-video compiled-adjoint
  replacement artifact, and shared-work evidence. It maps
  these to thirty-four proved requirements and deliberately keeps `full_goal_completion`
  open. New Q2 shared-work evidence compares one `Q2 x Omega x T` chart against
  replaying one `Omega x T` chart per q-pair: shared payload growth is `1.0x`,
  replay payload growth is `64.0x`, final payload ratio is `0.0625`, final
  chart ratio is `0.015625`, and max UV fit residual is `0.111px`. Focused
  Q2 shared-work plus goal-progress tests pass `46 passed in 5.78s`, and the
  saved goal-progress artifact verifies with `--verify-current-inputs`.
- The real-video acceptance envelope now lives at
  `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json`.
  It verifies twelve underlying source-distinct functional, frame-scaling,
  frame-count-breadth, quality-tether, and media-tether reports; covers five
  functional/media scenes, 10 broad media/quality sources, and four real-video
  frame-count points; keeps support rebins and stale refreshes at zero; keeps
  max rebuild ratio at `0.5`; preserves the expected strict timing failures;
  and records `does_not_prove_completion=true`. The updated goal-progress
  contract is covered by the timing-variance gate below.
- The real-video timing-variance envelope now lives at
  `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_variance_envelope/summary.json`.
  It consolidates the strict five-source timing misses, render-forward
  residual/shape diagnostics, Bq4 traced reruns, repeat/sequence/policy-order
  stability probes, and fresh-process isolation. The report keeps all timing
  misses cache/support clean, shows workload changes explain zero
  render-forward misses, verifies the traced Bq4 spike did not reproduce,
  passes fresh-process median acceptance with median no-first ratio
  `0.5645123618278631`, and records `does_not_prove_completion=true`. Focused
  timing-variance plus goal-progress tests pass `51 passed in 6.34s`.
- The goal-completion gap report now lives at
  `outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json`.
  It turns the remaining `full_goal_completion` opening into a machine-checked
  evidence-gap contract: broad real-scene quality acceptance, timing
  acceptance, and full compiled-adjoint trainer replacement are now proved by
  explicit artifacts inside this gap report. A broad10 real-video
  trainer matrix at
  `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10/summary.json`
  covers 10 distinct sources and 20 cadence/measured rows with all rows passing,
  exact cadence-loss matching, max rebuild ratio `0.5`, and zero support rebins
  or stale refreshes. A broad10 quality tether at
  `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_quality_tether/summary.json`
  verifies 10 source-distinct cadence/measured quality pairs, matching loss and
  RGB-loss curves within `2e-8`, all gradient flags present, positive PSNR
  gains, and min measured PSNR gain `0.03675997257232666`. A broad10 media
  tether at
  `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_media_tether/summary.json`
  verifies 10 source-distinct cadence/measured media pairs through the actual
  contact-sheet writer: max contact-sheet pixel delta `0`, matching PNG
  hashes, positive target/pred row stds, all gradient flags present, zero
  overflow/fallback/visibility stratification, rebuild ratio `0.5`, and scalar
  loss/PSNR deltas within the explicit media float-tick tolerances. A
  frame-count breadth diagnostic at
  `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_frame_count_breadth_diagnostic/summary.json`
  accepts the 4/8/16/32 frame-count coverage from the 4-count multiscene
  frame-scaling matrix as breadth evidence while preserving its strict timing
  failure as timing evidence. A timing-protocol acceptance artifact at
  `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance/summary.json`
  promotes the fresh-process median protocol with warmup discard as the timing
  acceptance contract: post-warmup median no-first/projective-total/
  feature-state-update ratios are
  `0.5645123618278631`/`0.8356591487478802`/`0.846418513757801`, strict
  warm-state misses stay diagnostic caveats, cache/support/workload invariants
  are clean, and `timing_acceptance_gap=0`. The completion-gap report now reduces
  `compiled_trainer_replacement_gap`,
  `compiled_trainer_source_gap`, `broad_quality_source_gap`,
  `broad_media_source_gap`, `broad_quality_frame_count_gap`,
  `strict_timing_failure_gap`, and `timing_acceptance_gap` to `0`;
  `open_gap_ids=["full_goal_completion"]`.
  It keeps `completion_ready=false` and `does_not_prove_completion=true`;
  focused compiled-adjoint/gap tests pass `23 passed in 4.45s`.
- The final completion audit now lives at
  `outputs/benchmarks/2026-05-25_star_uvt_projective_goal_final_completion_audit/summary.json`.
  It consumes the current goal-progress and completion-gap artifacts, verifies
  both against current inputs, checks the ten numbered theory subfolders and
  `GOAL_META_KEY_MATH.md`, and proves nine objective-level rows:
  theory/memory, fiber-bundle trace math and derivatives, revolving
  camera-family visibility atlas, Metal forward/backward renderer path,
  exposure/rolling/fallback contract, sublinear world-side bandwidth/work,
  broad real-video renderer acceptance, compiled-adjoint training replacement,
  and final completion promotion. It records
  `final_goal_completion_accepted=true`, `completion_ready=true`,
  `does_not_prove_completion=false`, and no open objective rows.
- The real-video compiled-adjoint replacement artifact now lives at
  `outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json`.
  It verifies the source-level renderer-adjoint path: trainer route to
  `_render_projective_interval_feature_tubes_autograd`, harness
  `_ProjectiveCellIntervalBackward`, interval Metal forward
  `render_projective_trace_cell_interval_atlas_metal`, and interval Metal
  direct VJP `direct_backward_projective_trace_cell_interval_atlas_metal`,
  with visibility order and tile membership treated as compiled constants. It
  also verifies 20 broad10 case payloads, all projective-interval main path,
  all RGB direct-loss autograd, all renderer gradient flags present, positive
  forward/backward timing, measured cache reuse, zero fallback/support churn,
  10 broad trainer sources, 10 broad quality/media sources, four frame-count
  points, and shared-work ratios below threshold. It records
  `final_compiled_adjoint_replacement_accepted=true` and
  `compiled_trainer_replacement_gap=0`; scope note: this is the practical
  direct-atomic RGB trainer route backed by the compiled interval adjoint, not
  deterministic compact static-STAR promotion.
- Finite-exposure and rolling-shutter semantics now have a focused saved
  artifact:
  `outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_quadrature/summary.md`.
  It locks the rule that exposure integrates the rendered sensor-time field,
  not pre-composited primitive opacity: finite exposure lowers into one shared
  interval atlas with exact CPU parity, rolling shutter deduplicates row sample
  times with a `0.875` unique/row-sample ratio, and all four available Metal
  cases match the CPU oracle within `5.96e-8`. The finite and rolling mixed
  fallback cases both mark `visibility_ambiguous_depth` cells at `0.5`
  fallback fraction and patch them against live-depth reference ordering.
- Finite-exposure and rolling-shutter backward semantics now have a matching
  saved artifact:
  `outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_backward/summary.md`.
  It pushes final-image adjoints back to sample adjoints with quadrature
  weights or `row_weights`, then calls one shared interval-cell Metal VJP.
  Against Torch autograd on the lowered interval atlas, finite and rolling
  Metal gradients match with max absolute error `1.43e-6` and max relative
  error `6.38e-7`; rolling still reuses the deduplicated `7/8` sample schedule.
- Visibility-ambiguous finite/rolling fallback now has an explicit
  differentiable backward artifact:
  `outputs/benchmarks/2026-05-25_star_uvt_projective_exposure_rolling_mixed_fallback_backward/summary.md`.
  It renders non-fallback regions through the trainer-harness interval Metal
  VJP, patches `visibility_ambiguous_depth` tile/sample regions with live-depth
  Torch reference gradients, then applies exposure or row weights. On this MPS
  run both finite and rolling mixed backward cases pass with max output error
  `5.96e-8`, max gradient absolute error `2.15e-6`, max gradient relative
  error `7.41e-7`, and rolling row-time reuse `11/12`.
- For alpha/background regularization, the 20-step 8f/64px smoke favored
  `random_feature_before_colorizer`, but the longer 100-step same-shape rerun
  flips the ordering: `random_rgb_after_colorizer` wins fixed-black eval in
  both dynamic gsplat (`15.915` PSNR vs `10.917`, alpha mean `0.890` vs
  `0.337`) and STAR UVT (`20.897` PSNR vs `20.662`). The 100-step run lives at
  `outputs/benchmarks/2026-05-21_alpha_background_ablation_100step_212901/summary.md`;
  higher-res 100-step confirmations now live at
  `outputs/benchmarks/2026-05-21_alpha_background_ablation_128px_100step_224500/summary.md`
  and
  `outputs/benchmarks/2026-05-21_alpha_background_ablation_256px_100step_225800/summary.md`.
  At 128px the signal is mixed: dynamic random RGB has lower eval loss and
  higher alpha while random feature background has higher PSNR; STAR random RGB
  has slightly better quality but random feature is faster. At 256px, dynamic
  clearly favors random RGB (`16.350` vs `15.035` PSNR, alpha `0.894` vs
  `0.796`), while STAR clearly favors random feature background (`15.889` vs
  `12.757` PSNR). Treat background choice as renderer/scale-specific, not a
  global default.
  the older short-run artifacts remain at
  `outputs/benchmarks/2026-05-21_alpha_background_ablation/summary.md`,
  `outputs/benchmarks/2026-05-21_alpha_background_ablation_confirm/summary.md`,
  `outputs/benchmarks/2026-05-21_alpha_background_ablation_latest/summary.md`,
  and
  `outputs/benchmarks/2026-05-21_alpha_background_ablation_refresh_210512/summary.md`.
  For the next scale run, set the policy per renderer and rerun at the actual
  target resolution before 300-video promotion.
- The first direct feature Metal op is intentionally simple: emit dense
  feature image plus alpha, accept `grad_feature_image` plus `grad_alpha`, and
  use direct atomic gradients for geometry/opacity/features.
- The compatible projective interval cache-policy artifact now lives at
  `outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step/summary.md`.
  It is a narrow `feature_dim=3`, full-frame route, not the future F32
  target-grid endpoint. Measured policy reduces full atlas rebuilds `4 -> 1`,
  keeps final loss identical (`0.0847767964`), and improves no-first-step mean
  step time `3473.3 -> 2137.2ms`, but support still rebins on every live
  update. Next projective-cache work should reduce metadata invalidation under
  ordinary tube motion before claiming the cache policy is solved.
- The first support-guard projective interval artifact lives at
  `outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_cap256/summary.md`.
  `support_guard_padding=2` with `STAR_UVT_TILE_CAPACITY=256` preserves the
  same final loss (`0.0847767964`) while eliminating stale refreshes and support
  rebins (`4/7 -> 0/0` for cadence/measured). Measured still reduces full
  atlas rebuilds `4 -> 1` and improves no-first-step mean time
  `7468.6 -> 2496.7ms`. The negative side is important: guard `2` and guard
  `8` both overflow at the old cap128. Treat support guards as a budgeted chart
  margin, not a free default.
- The first cap-aware guard policy is now implemented as
  `projective_interval.support_guard_policy="budgeted"`. The cap128 artifact
  `outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_budgeted_cap128/summary.md`
  avoids the immediate packed-tile overflow and the measured row passes with
  the same final loss, zero tile overflow, and rebuilds still at `1`, but it is
  mostly a negative performance/churn result: support rebins only improve
  `7 -> 6`, no-first-step mean slows to `6107.7ms`, and the cadence row times
  out because repeated budget searches are expensive. Next support-guard work
  should be local/headroom-aware per tile/trace, not global bisection.
- The first local headroom guard policy is now implemented as
  `projective_interval.support_guard_policy="local_budgeted"`. It compiles the
  full guard, detects only packed tiles that overflow, and replaces those tiles
  with base-support cells while preserving guarded cells elsewhere. The explicit
  cap128 artifact
  `outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_local_budgeted_cap128_explicit/summary.md`
  passes both cadence and measured rows with identical final loss
  (`0.0847767964`) and zero tile overflow (`max_tile_count=70` in the case
  JSONs). It removes the global-bisection timeout and measured no-first-step is
  back to `2468.3ms`, but it still rebins support on every live update
  (`7/7`). Treat this as the right cap-safety structure, not the final churn
  fix; the next guard must allocate headroom per trace/cell inside crowded
  tiles or split/refit the local offenders.
- The first trace-headroom policy is now implemented as
  `projective_interval.support_guard_policy="trace_budgeted"`. It keeps
  base-active trace ids in overflowing tiles and spends remaining tile capacity
  on deterministic extra guarded trace ids before falling back to tile-local
  base support. The explicit cap128 artifact
  `outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_rerun/summary.md`
  passes both rows with identical final loss and zero overflow
  (`max_tile_count=70`), but still does not solve stale support churn:
  measured remains `7/7` support rebins with no-first-step `2460.0ms`. This
  weakens the hypothesis that slot allocation alone is enough; next look at
  motion-aware guard sizing/support-event roots or cap256 as the honest churn
  win.
- Support-boundary overshoot telemetry and a bounded stale-overshoot debounce
  now exist. The trace-budgeted cap128 margin artifact
  `outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_margin/summary.md`
  shows the old `7/7` measured rebins were triggered by tiny support-boundary
  crossings (`max_support_max_overshoot_px=0.0912`). With
  `support_stale_overshoot_epsilon=0.125`, measured rebins fall `7 -> 3`; with
  `0.25`, they fall `7 -> 1`; with `0.5`, the artifact
  `outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_trace_budgeted_cap128_eps05/summary.md`
  reaches `0/7` measured support rebins, zero overflow, and the same final loss
  (`0.0847767964`) with no-first-step `1277.5ms`. Image-level tests now bound
  the tolerance on both an axis-aligned boundary case and a tiny rational
  orbit-derived trace: real support padding keeps the strict-rebinned versus
  tolerant-reused max RGB error below `1e-4`, while underspecified center-only
  support still exceeds `0.35`. Treat `0.5px` as a bounded experimental
  debounce, not a universal default, until broader scene/error tolerance is
  checked.
- Support debounce now has a math certificate, not just a pixel-distance knob.
  `projective_interval.support_stale_tail_alpha_epsilon` lets measured refresh
  reuse stale support only when omitted Gaussian tail opacity is below budget.
  The bound now aggregates omitted tails per missing sample/tile instead of
  taking a max over primitives, so overlapping low-alpha support loss cannot be
  hidden by a per-trace certificate. Focused tests show a real-support `0.05px`
  sliver is debounced when the tail-alpha budget is `3e-4`, rejected when the
  budget is `1e-4`, center/core loss with `uv_padding=0` still rebins because
  the bound is `0.5`, and 16 overlapping tiny tails rebin at `1e-3` because
  their aggregate bound is about `0.00327`. The benchmark CLI now exposes
  `--support-stale-tail-alpha-epsilon`, and summaries include
  `projective_interval_cache_last_support_tail_alpha_bound` plus
  `projective_interval_cache_max_support_tail_alpha_bound`, so cache reuse can
  be audited against the certificate in real artifacts.
- The old max-per-trace tail-alpha artifacts are superseded. After aggregate
  accounting, the compatible slack-budgeted cap128 smoke is path-dependent:
  `0.00035` now records `2` stale refresh/support rebins with max aggregate
  bound `0.000404648`;
  `outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00035_aggregate/summary.md`.
  `0.00045` and `0.0006` still record one support rebin as skipped earlier
  repairs allow later drift to grow (`0.000526049` and `0.000656625` max
  aggregate bounds). The corrected `0.001` artifact clears this smoke with
  identical loss, zero overflow, one rebuild versus cadence's four, seven live
  updates, zero stale refreshes/support rebins, and max aggregate tail bound
  `0.000736007`:
  `outputs/benchmarks/2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail001_aggregate/summary.md`.
- The first tail-alpha image-error verifier is now recorded at
  `outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error/summary.md`.
  It compares strict support rebinning against certified reuse on three affine
  boundary-tail cases plus one tiny rational orbit chart. All four reuse cases
  pass with max RGB error below the omitted-alpha bound: worst positive case is
  `axis_r6_sigma15_opacity09` with tail bound `0.0003447873` and max error
  `0.0000822361`; the orbit case has tail bound `0.0002094069` and max error
  `0.0000227757`. The negative `uv_padding=0` core-loss case rejects
  tail-certified reuse (`tail_bound=0.5`), while a pixel-only overshoot pardon
  would have produced max RGB error `0.3987594`. Treat this as a local
  isotropic/projective verifier that strengthens the certificate, not yet a
  broad anisotropic-scene guarantee.
- The aggregate tail-alpha verifier is recorded at
  `outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_tail00035_aggregate/summary.md`.
  At `tail_alpha_epsilon=0.00035`, the three single-tail cases and rational
  orbit case still reuse with max RGB error below their tail bounds, while the
  64-trace overlapping-tail negative rejects reuse with aggregate bound
  `0.01309515`; forcing that reuse would produce `0.00141417` max RGB error.
- The next certificate-math gate now covers anisotropic screen footprints at
  the CPU/theory level:
  `outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound/summary.md`.
  The verifier minimizes the SPD Mahalanobis quadratic over each omitted tile
  rectangle, then sums omitted alpha from traces landing in the same missing
  tile. It passes a diagonal anisotropic tail (`bound=0.0002046116`,
  max error `0.0000242851`), a rotated precision tail (`bound=0.0001845283`,
  max error `0.0000190703`), and a two-trace same-tile sum
  (`bound=0.0002287796`, max error `0.0000166098`). The anisotropic core-loss
  negative rejects reuse (`bound=0.5`) and shows the omitted core would have
  max error `0.4379515`. This is the math shape needed for q-UVT / gauged
  footprints; the interval atlas now carries per-trace footprint precision.
- `ProjectiveTraceCellTraceAtlas` now carries optional
  `spatial_precision_uv` metadata with shape `[N,3]` for `(q_uu, q_uv, q_vv)`.
  Metadata validation rejects non-SPD precision blocks, support/visibility
  atlas transforms preserve it, q-UVT atlas lowering and live-atlas updates
  populate it from `q_uvt`. The CPU/reference cell renderer, finite-exposure
  quadrature reference path, and stale-support tail-alpha certificate now
  consume this precision as the local quadratic footprint. Production
  projective interval Metal forward/backward now consume the same fixed
  per-trace precision; scalar `sigma_px` is used only to synthesize isotropic
  precision when metadata is absent. The q-UVT compatibility lowering path is
  isotropic by default, but now has an explicit anisotropic opt-in that carries
  the UV precision block and expands support by the alpha-threshold ellipse
  bound. The learned source-view trainer route still locks q-UVT precision to
  isotropic until that model/init contract changes. Focused precision/Metal
  tests plus the broad projective/interval suite pass (`143 passed in
  16.99s`). Current-code
  verifier reruns are
  `outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_metal_precision_rerun/summary.md`
  and
  `outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound_metal_precision_rerun/summary.md`.
  The scalar rerun also includes an overlapping-tail aggregate rejection:
  same-tile omitted tails sum to `0.01309515`, exceed the `0.001` budget, and
  correctly prevent reuse.
- `ProjectiveTraceCellTraceAtlas` now also carries optional
  `depth_affine_uv` metadata with shape `[N,6]` for a pixel-varying conditional
  depth plane `z(u,v,t) = z_c(t) + z_u(t)(u-u_c(t)) + z_v(t)(v-v_c(t))`.
  The Torch helper `eval_projective_trace_cell_depth_at_uv_torch(...)`
  evaluates this screen-fiber depth model, validation rejects malformed slope
  tensors, and support rebinning, trainer refresh, quadrature lowering, and
  CPU detach preserve it. Production interval Metal forward/rows/backward now
  receive `depth_affine_uv` and evaluate the pixel depth plane inside dynamic
  selection sorting, with zero slopes preserving the legacy scalar-depth path.
  The depth-plane slopes remain compiled metadata, not a trainable VJP target.
  Targeted Metal depth-affine/spatial-precision tests pass (`3 passed in
  3.18s`), and the broad projective/interval suite passes (`149 passed in
  19.91s`).
- The interval Metal direct backward now returns `grad_spatial_precision_uv`
  for the smooth anisotropic footprint metric. The VJP uses
  `d alpha / d q_uu = -0.5 alpha du^2`,
  `d alpha / d q_uv = -alpha du dv`, and
  `d alpha / d q_vv = -0.5 alpha dv^2`; the trainer-harness interval autograd
  path passes `spatial_precision_uv` as a differentiable input when present.
  Direct backward matches Torch autograd for spatial precision, and an
  anisotropic q-UVT opt-in route now backprops into `q_uu/q_vv`. Focused tests
  pass (`2 passed in 4.17s`), and the broad projective/interval suite passes
  (`152 passed in 16.41s`).
- The source-view projective interval trainer now respects
  `allow_anisotropic_spatial_precision`: default configs still lock
  `raw_precision[:,0:2]` to `sigma_px^{-2}` and mask those gradients, while
  the anisotropic opt-in skips the lock and lets the interval Metal VJP train
  spatial precision. Focused locked/unlocked bridge tests pass (`2 passed in
  8.76s`), py_compile passes, and the broad projective/interval suite passes
  (`153 passed in 20.31s`).
- The source-view tube model now has an SPD-safe trainable UV cross precision
  parameter, `raw_spatial_correlation`, so the q-UVT footprint can learn a
  rotated screen ellipse rather than only axis-aligned `q_uu/q_vv`. The
  parameterization uses
  `q_uv = rho_max * tanh(raw_spatial_correlation) * sqrt(q_uu*q_vv)` with
  `rho_max < 1`, so `q_uu*q_vv - q_uv^2 > 0` by construction. The default
  projective interval lock zeros and masks this route; the anisotropic opt-in
  leaves it trainable and the interval Metal VJP propagates cross-term
  gradients under an asymmetric pixel loss. Focused cross/locked/unlocked tests
  pass (`3 passed in 4.90s`), py_compile passes, and the broad
  projective/interval suite passes (`154 passed in 23.49s`).
- The compatible projective interval trainer now has a tiny frame-scaling
  artifact through the actual `run_training` route:
  `outputs/benchmarks/2026-05-24_star_uvt_projective_interval_trainer_frame_scaling/summary.md`.
  It monkeypatches the video loader with synthetic tensors but leaves the real
  feature trainer/projective-interval path in charge. For `4/8/16` frames and
  four optimizer steps, cadence rebuilds stay `2/2/2`, measured policy keeps
  rebuilds at `1/1/1`, all rows pass, all rows have zero tile overflow,
  measured end loss matches cadence exactly (`max delta 0.0`), and the
  measured path preserves live cache updates plus staleness checks. Treat the
  timing columns as smoke-level diagnostics only; the durable claim here is
  real trainer-route cache reuse with matching loss and bounded tile pressure.
- The compatible projective interval trainer now also has a real-video
  frame-scaling artifact on the checked-in high-motion smoke clip:
  `outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling/summary.md`.
  It runs actual `run_training` cases for `4/8/16` frames at 64px/128 tubes,
  comparing cadence rebuilds with the measured live-cache policy. Cadence does
  `2/2/2` full cache rebuilds; measured does `1/1/1` while preserving
  identical end loss (`max delta 0.0`), zero tile overflow, live updates, and
  staleness checks. Measured no-first-step ratios versus cadence are
  `0.305/0.394/0.743`. Support rebins still occur on every live update
  (`3/3/3`), so this is trainer-level cache-reuse evidence, not the final
  no-churn support-lifecycle result.
- Guarded real-video trainer reruns now live at
  `outputs/benchmarks/2026-05-24_star_uvt_projective_real_video_trainer_frame_scaling_guard025_tail001/summary.md`,
  `..._guard05_tail001/summary.md`, `..._guard1_tail001/summary.md`, and
  `..._guard2_tail001/summary.md`. They use
  `support_guard_policy=slack_budgeted` and
  `support_stale_tail_alpha_epsilon=0.001`. On the same high-motion `4/8/16`
  rows, all four guards keep measured rebuilds at `1/1/1`, match cadence end
  loss exactly, keep zero overflow, and eliminate support rebins/stale
  refreshes (`0/0/0`). Guard `0.25px` is the smallest certified no-churn
  guard; guard `0.5px` has the best measured/cadence no-first-step ratios in
  this tiny ladder (`0.536/0.324/0.279`). Guard `2px` also clears rebins but
  regresses the `16f` measured row to `1.753x` cadence, so guard cost is a
  real tuning axis rather than a free fix.
- The first trained high-motion trace-geometry scaling artifact now lives at
  `outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.md`.
  It trains a tiny projective-interval STAR UVT feature model on the checked-in
  high-motion smoke video, saves/reloads the checkpoint, and compiles both
  init and trained tensors into projective interval atlases over `4/8/16`
  frame prefixes. The trained checkpoint row reduces loss
  `0.298236 -> 0.296121`, passes with zero tile overflow, keeps fallback
  fraction `0.0`, and keeps trace count fixed at `64`; dense per-frame tile
  pairs grow `1542 -> 6016` while interval trace entries grow only
  `392 -> 573`, so the trained interval/dense tile-pair ratio drops
  `0.254 -> 0.095`. The regenerated artifact now includes a repeated
  per-frame interval baseline on the same learned checkpoint. At `16` frames,
  the shared interval row uses `573` interval entries versus `3862` for
  per-frame atlas replay; warm tiny-MPS forward/backward timing is
  `57.0/60.3ms` for the shared row versus `355.2/367.9ms` for the per-frame
  row. This replaces the old motion-centroid-only evidence with a real saved
  trainer-checkpoint geometry gate. Treat timing as small diagnostic evidence,
  not final high-resolution wall-clock proof.
- The larger trained high-motion smoke artifact now lives at
  `outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.md`.
  It repeats the same saved-checkpoint interval-vs-per-frame replay gate at
  64px and 128 tubes. The trainer row decreases loss
  `0.317323 -> 0.316218`, with one cache rebuild, three live updates, zero
  tile overflow, and zero fallback. For the trained checkpoint over `4/8/16`
  frames, dense per-frame tile pairs grow `3578 -> 14363` while shared
  interval entries grow only `956 -> 1371`; the interval/dense ratio drops
  `0.267 -> 0.095`. The same learned tensors replayed framewise need `9605`
  interval entries at `16f`, versus `1371` for the shared interval atlas.
  Tiny warm MPS timing at `16f` is `469.7/303.3ms` forward/backward for shared
  interval versus `802.0/1779.1ms` for per-frame replay. The forward row is
  noisier than the 32px artifact, but the backward and entry-count evidence
  strengthens the claim that the reusable sensor-time atlas survives a modest
  scale-up.
- The next trained high-motion smoke artifact now lives at
  `outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.md`.
  This run exposed and fixed a benchmark-harness bug first: non-default
  `tile_capacity=256` must be pushed into `STAR_UVT_TILE_CAPACITY` before the
  trainer render, not only before the standalone timing kernels. The focused
  verifier test now locks that env synchronization. With the fix, the 96px/256
  tube cap256 run passes the saved-artifact contract: loss decreases
  `0.317038 -> 0.315874`, cache rebuild/live/stale remains `1/3/3`, and
  overflow/fallback stay zero. For the trained checkpoint over `4/8/16`
  frames, dense per-frame tile pairs grow `7820 -> 31255` while shared
  interval entries grow only `2045 -> 2831`; the interval/dense ratio drops
  `0.262 -> 0.091`. At `16f`, per-frame replay uses `20547` interval entries
  versus `2831` shared, and warm tiny-MPS timing is `250.6/247.2ms` for shared
  forward/backward versus `2512.6/2018.8ms` for per-frame replay.
- The variable-camera orbit segment producer now has an explicit revolving
  camera fiber-metric gate. A synthetic elevated look-at orbit with anisotropic
  world tubes compiles `16` frames into `4` temporal chart segments per tube
  instead of one segment per frame; those charts carry nonzero SPD `q_uv`
  cross precision, and the first tube's cross term changes sign across the
  orbit. The same projected segments render finite nonzero CPU UVT output.
  Focused orbit tests pass (`7 passed in 2.18s`), py_compile passes, and the
  broad projective/interval suite passes (`158 passed in 14.19s`).
- The revolving-camera orbit fixture now has a chart-size error/share sweep
  against the one-segment-per-frame route. For `8` frames, chart sizes
  `1/2/4/8` produce segment ratios `1.0/0.5/0.25/0.125`; all shared routes
  stay below `0.009` mean absolute image error, `0.0011` MSE, and `0.40`
  max absolute error against the framewise reference. Focused sweep passes
  (`1 passed in 10.70s`), the full orbit file passes (`8 passed in 14.11s`),
  py_compile passes, and the broad projective/interval suite passes
  (`160 passed in 26.11s`).
- The revolving-camera chart-size sweep now lowers each route into the
  projective interval atlas and checks interval compression/fallback stats.
  For `frames_per_segment = 1/2/4/8`, trace counts are `16/8/4/2`, fallback
  fraction stays `0.0`, and interval-to-dense trace-sample ratio decreases
  from `1.0` to below `0.35`; atlas reference output remains within `0.02`
  max absolute error and `3e-5` mean absolute error of the charted UVT render.
  The orbit interval Metal path also matches the reference on the same family.
  Focused interval tests pass (`2 passed in 10.42s`), the full orbit file
  passes (`11 passed in 9.96s`), py_compile passes, and the broad
  projective/interval suite passes (`163 passed in 33.16s`).
- The revolving-camera interval atlas route now has an orbit-derived backward
  gate. Differentiable chart tensors are lowered into the interval atlas and
  rendered through the Metal autograd bridge; an asymmetric image loss
  backprops into the chart centers, opacity, color, and all six `q_uvt`
  entries, including the rotated `q_uv` cross precision. Focused backward
  passes (`1 passed in 3.37s`), the full orbit file passes (`12 passed in
  18.07s`), py_compile passes, and the broad projective/interval suite passes
  (`164 passed in 28.66s`).
- The revolving-camera orbit lane now has a fixed-chart-count frame-growth
  gate. With `frames = 8/16/32` and a fixed four temporal charts per tube,
  per-frame segment counts grow `16/32/64`, but compiled chart segments and
  atlas trace counts stay `8/8/8`. Dense trace samples grow `156 -> 820`,
  while interval trace entries only grow `99 -> 156`, and the interval ratio
  drops `0.635 -> 0.190` with zero fallback. This is the current executable
  certificate for the goal condition: output pixels still grow, but world-side
  projection/support/atlas work is shared over the orbit. Focused frame-growth
  gate passes (`1 passed in 12.13s`), the full orbit file passes (`13 passed in
  33.10s`), py_compile passes, and the broad projective/interval suite passes
  (`165 passed in 48.65s`).
- The revolving-camera interval backward route now has a fixed-chart-parameter
  frame-densification gate. With `frames = 4/8` and a fixed two temporal charts
  per tube, segment counts and atlas trace counts stay `4/4`; the Metal
  interval autograd path still backprops nonzero gradients into chart centers,
  opacity, color, the rotated UV precision block including `q_uv`, and temporal
  `q_uvt` terms. Focused backward gates pass (`2 passed in 4.28s`), the full
  orbit file passes (`14 passed in 30.73s`), py_compile passes, and the broad
  projective/interval suite passes (`166 passed in 49.64s`).
- The first measured revolving-camera fixed-chart scaling artifact now lives at
  `outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.md`.
  It compares fixed four-chart-per-tube orbit compilation against one chart per
  frame for `4/8/16/32` output frames. Fixed-chart rows keep segment/trace
  counts `8/8/8/8`, constant atlas tensor payload `608` bytes, zero fallback,
  interval entries `112/99/135/156`, and interval ratio `1.0 -> 0.190`; the
  per-frame route grows segment/trace counts `8/16/32/64` and atlas payload
  `608 -> 4864` bytes. The regenerated artifact now also records CPU
  projection/atlas-build compile timing: fixed-chart compile grows
  `27.80 -> 36.64ms` (`1.32x`) while per-frame compile grows
  `21.93 -> 296.22ms` (`13.51x`), making the `32f` fixed route `0.124x` the
  per-frame compile cost. On the same small prewarmed MPS diagnostic,
  fixed-chart forward/backward is `0.153x/0.267x` the per-frame route at `32f`.
  Treat timing as a synthetic diagnostic, but the count/payload/autograd and
  compile-side evidence are now saved as a repeatable artifact.
- `feature_uvt.render_mode=feature_direct_fixedbin` is currently a mode
  contract around the direct feature path: it records requested/effective mode
  and requires fallback when tile overflow appears. Do not treat it as a
  separate optimized fixedbin kernel until the Metal backward actually changes.
- `feature_uvt.render_mode=feature_direct_gradcache` is the first real
  feature-backward fast mode. It caches each pixel's F32 grad vector in the
  direct backward kernel when `feature_dim <= 64`.
- `gradcache_skip_feature_grad` is a benchmark-only diagnostic. It intentionally
  zeros feature gradients while preserving geometry/opacity gradient parity, so
  it can isolate the feature-gradient atomic cost. Do not put it in train
  configs.
- `feature_direct_gradcache_reduce` / `gradcache_reduce_feature_grad` is a
  trainable reduced-feature-gradient prototype. It preserves gradients and
  passes parity/training, but the current per-contributor threadgroup reduction
  is slower than plain gradcache on the target row, so it is not promoted.
- `feature_direct_gradcache_reduce_vec4` /
  `gradcache_reduce_feature_grad_vec4` is the vectorized follow-up. It reduces
  feature channels in `float4` groups and passes F4/F32 parity, but it is only
  a synthetic direct-kernel win; the first-class cap256 real-video row is
  slower than gradcache and scalar reduce, so it is not promoted.
- The first matched 64f/512px target-grid/frozen-probe trainer render-mode
  matrix confirmed that this remained true under dense analytic VJP:
  `feature_direct_atomic`, gradcache, cached-bins, scalar reduce, vec4 reduce,
  and the fixedbin request all pass from the same 1300-step checkpoint with the
  same final loss/probe PSNR, but vec4/reduce do not win end-to-end. The
  repeat-top row has direct-atomic no-first `1249.0ms`, cached-bins `1410.9ms`,
  vec4 `1509.6ms`, and fixedbin-request `1422.6ms`. Fixedbin-request reports
  `kernel_backward_mode=direct_atomic`; it is not a native fixedbin kernel. The
  later sparse-grid matrix and sparse-forward trainer gate below supersede this
  dense-analytic matrix for the selected speed path.
- The first-class 512px/8192t no-pre-norm RGB-target rerun is the exception for
  older RGB speed diagnostics: `feature_direct_gradcache_reduce_vec4` matches
  gradcache quality and improves the 20-step media row from `2.858s/step`,
  `1.327s` backward to `2.491s/step`, `1.184s` backward with zero overflow.
  Keep it as `star-feature-512-rgbfast`, not as the current V-JEPA target route
  or as a quality baseline.
- The selected-shader 128/256/512 scale gate now bounds that decision:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_selected_shader_scale_128_256_512.md`.
  Vec4 reduce is a meaningful 512px win, only a small 256px win, and a
  tie/slight backward loss at 128px. The 128px row also required cap256 plus
  `alpha>=1/32`; cap128/default alpha, cap256/default alpha, and
  cap256/`alpha>=1/72` all overflowed.
- The original selected fast feature route was not a cached V-JEPA target
  route; the bridge audit showed the old config had no `features` section and
  still loaded RGB video targets through `FeatureToColor`. That is now
  superseded by `star-feature-512-fast`, which launches the cached V-JEPA
  target-grid/frozen-probe sparse-forward batched VJP route. The old RGB-target
  speed row remains available as `star-feature-512-rgbfast`.
- The first STAR target-feature bridge smoke now passes with `rgb_pyramid`.
  `feature_target.enabled=true` loads cached `rgb_x1` via `VideoFeatureCache`,
  adapts it to the rendered F32 grid with `repeat_truncate`, disables RGB loss
  (`rgb_loss_weight=0`), and trains on `render.feature_image`. The cache-hit
  8f/64px/512t/F32 rerun passes at loss `0.34006 -> 0.24809`, mean
  `93.5ms/step`, `43.0ms` backward, and zero tile overflow. This proved the
  cached-target contract before the real V-JEPA smoke below.
- The real V-JEPA target smoke now passes. `vjepa_torchhub`/`vjepa2_1_vit_base_384`
  returns `vjepa_tokens` as `[1,1024,768]`; STAR uses
  `feature_target.token_grid_shape=[4,16,16]`, `truncate_or_pad`,
  `trilinear`, and `channel_standardize` to train directly against
  `render.feature_image`. The 8f/64px/512t/F32 cache-hit row passes at loss
  `1.00082 -> 0.90042`, mean `181.1ms/step`, `53.8ms` backward, and zero tile
  overflow. This is a bridge smoke, not the selected 512px scale/quality gate.
- The real V-JEPA target 512px scale gate now passes with chunked target
  materialization. The 64f/512px/8192t/F32 cache-hit row uses
  `feature_direct_gradcache_reduce_vec4`, adapts `vjepa_tokens` `[1,8192,768]`
  with `token_grid_shape=[32,16,16]` to a channel-adapted `[32,32,16,16]`
  source, and streams logical `[64,32,512,512]` target chunks. It passes at
  loss `1.000014 -> 0.999545`, `3.743s/step`, `1.077s` backward,
  `1.734s` target chunk/loss, and zero tile overflow. The first attempt failed
  with a 48 GiB interpolation temporary; channel adaptation now happens before
  grid upsampling, and chunking avoids a resident full dense target. Treat this
  as scale/trainability evidence, not a quality baseline.
- The cached-target-layout follow-up for that same V-JEPA gate now passes too.
  `feature_target.materialization=cached_chunks` precomputes the adapted target
  into 32 resident chunks (`2048MiB`, `2.044s` load/prep) and cuts the cache-hit
  5-step row to `1.655s/step`, `0.770s` backward, `0.601s` render, and
  `0.202s` target chunk/loss with the same loss curve and zero tile overflow.
  This is the exact dense render-grid-loss reference. The target-cache budget
  gate quantifies the cliff: float32 adapted targets are `4GiB` at 128f/512px/F32
  or 64f/512px/F64 and `8GiB` at 64f/1024px/F32.
- The target-grid V-JEPA loss follow-up now passes and is the current
  speed/memory diagnostic. `feature_target.materialization=target_grid` keeps
  only the channel-adapted `[32,32,16,16]` V-JEPA grid resident (`1.0MiB`) and
  downsamples rendered feature chunks before loss. The same 64f/512px/8192t/F32
  5-step row passes at loss `0.999935 -> 0.999467`, `1.351s/step`, `0.705s`
  backward, `0.548s` render, `0.041s` target/loss, `0.138s` target load/prep,
  and zero tile overflow. It is faster and much smaller than cached chunks, but
  it changes the objective from dense render-grid MSE to coarse token-grid MSE;
  the 20-step media follow-up confirms monotonic target-feature overfit
  (`0.999935 -> 0.997425`) at `1.451s/step`, `0.722s` backward, `0.630s`
  render, and `0.037s` target/loss, but it is not RGB quality evidence because
  `rgb_loss_weight=0` and the colorizer is not trained.
- The RGB-aux1 target-grid probe now passes as the first visual-control row.
  It uses the same target-grid V-JEPA route with `rgb_loss_weight=1.0`, trains
  the colorizer (`colorizer_grad_seen=true`), and decreases both feature loss
  (`0.999935 -> 0.997336`) and RGB loss (`0.338171 -> 0.335263`). It costs
  `2.000s/step`, `1.114s` backward, `0.586s` render, and `0.052s` target/loss.
  The RGB PSNR gain is only `4.709 -> 4.746` in 20 steps, so this is not a
  quality promotion; use it to motivate a stronger/longer RGB auxiliary schedule
  or a trained/frozen feature-to-RGB probe.
- The RGB-aux10 target-grid probe is a weak negative control for "just raise
  the RGB weight." It passes and decreases RGB loss slightly more
  (`0.338171 -> 0.334961`, PSNR `4.709 -> 4.750`), but feature loss ends worse
  than aux1 (`0.997547` vs `0.997336`) and step time stays about `2.0s`. Use
  a trained/frozen feature-to-RGB probe next, not only a larger RGB scalar.
- The 100-step RGB-aux10 target-grid probe now passes and shows schedule length
  does matter. It reaches feature loss `0.964670` and RGB PSNR `5.109` from the
  same `4.709` start, at `1.876s/step`, `1.033s` backward, `0.580s` render, and
  `0.043s` target/loss with zero overflow. This is a positive schedule signal,
  not a quality promotion: it is still far below the same-clip RGB STAR bracket.
- The matched RGB-warm20 target-grid schedule is a negative visual-control
  gate. The trainer now supports `feature_target.weight_schedule`; the probe
  uses `feature=0/rgb=20` for 20 steps and then `feature=1/rgb=10` through step
  100. It passes and is cheaper (`1.639s/step`, `0.872s` backward), but ends
  worse than constant aux10 at the same step count: RGB PSNR `5.046` vs
  `5.109`, feature loss `0.973557` vs `0.964670`. Do not spend more turns on
  feature-loss-skipping warmup without a different decoder/probe hypothesis.
- The standalone target-grid feature-to-RGB probe now passes and changes the
  next bridge. Config
  `src/train_configs/star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.jsonc`
  trains only a hidden64 `FeatureToColor` on the cached `[32,32,16,16]` V-JEPA
  target grid. It reaches grid PSNR `23.401` and full-video upsampled PSNR
  `20.073` at `2.427ms/step`, with `1.003ms` backward and offline W&B
  `7nlur74e`. This proves the target-grid features are visually decodable; the
  missing piece is loading/freezing that decoder inside STAR objective/logging,
  not another RGB aux weight or warmup schedule.
- The frozen RGB-probe decoder is now wired into the STAR target-grid trainer.
  Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_20step_media.jsonc`
  loads the checkpoint above, freezes it, and adds `rgb_probe_loss_weight=10`
  at the token grid. It passes at `1.220s/step` with feature loss
  `0.999935 -> 0.998357`, frozen-probe PSNR `13.985 -> 14.060`, `0.572s`
  backward, and offline W&B `f7v5bs0r`. This is the integration/speed proof;
  20-step visual movement is too small to call quality promotion.
- The matched 100-step frozen RGB-probe STAR gate passes and was the first
  stronger visual diagnostic beyond the 20-step plumbing proof. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_100step_media.jsonc`
  uses the same hidden64 checkpoint and probe weight, with offline W&B
  `3f4hm6wq`. It reaches feature loss `0.970035` and frozen-probe PSNR
  `14.641` from the same `13.985` start at `1.268s/step`, `0.630s` backward,
  `0.532s` render, `0.017s` target-grid loss, and `0.031s` probe loss. This is
  cheaper than 100-step RGB-aux10 and moves visual-probe PSNR more, but it still
  does not close the standalone `20.073` PSNR oracle gap.
- The 300-step frozen RGB-probe extension keeps improving, so the probe objective
  is viable rather than just a short-run artifact. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_300step_media.jsonc`
  uses the same route and offline W&B `jhv2lgdj`. It reaches feature loss
  `0.811652` and frozen-probe PSNR `16.560` at `1.355s/step`, `0.681s`
  backward, `0.552s` render, `0.019s` target-grid loss, and `0.037s` probe
  loss. This is still below the standalone `20.073` oracle, but the remaining
  gap is now a real objective/training question rather than a decodability
  failure.
- The STAR feature overfit trainer now has an opt-in checkpoint/resume contract
  for longer gates: `output.checkpoint` saves model, colorizer, optimizer,
  serialized config, row, and losses; `train.resume_checkpoint` loads those
  states, and `train.resume_optimizer` defaults true. The semantics are
  warm-start local steps, not global-step schedule continuation. A real
  8f/64px RGB-pyramid runtime smoke wrote `/tmp/star_uvt_checkpoint_resume_smoke/first.pt`,
  resumed from it with optimizer state, wrote `/tmp/star_uvt_checkpoint_resume_smoke/resume.pt`,
  and passed with zero overflow (`159.2ms/step` first tiny run,
  `42.8ms/step` resumed cache-hot run).
- The checkpoint/resume path is now exercised at the real 64f/512px/8192t
  frozen-probe scale. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_300step_checkpoint.jsonc`
  reruns the 300-step keeper without media, writes
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step.pt`,
  and matches the prior curve at `1.268s/step`, feature loss `0.811652`, and
  probe PSNR `16.560`. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_resume300_from300_checkpoint_media.jsonc`
  resumes from that checkpoint for 300 more local steps, writes media and a
  600-step-after-resume checkpoint, and passes with feature loss
  `0.810827 -> 0.655366`, frozen-probe PSNR `16.576 -> 19.884`, zero overflow,
  `1.440s/step`, and offline W&B `vtti65kr`. This nearly reaches the standalone
  full-video upsample number (`20.073`). Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.jsonc`
  resumes the 600-step checkpoint with explicit `train.global_step_offset=600`
  and a probe-emphasis objective (`feature=0.25`, `rgb_probe=40`) for 200 more
  local steps. It passes with probe PSNR `19.888 -> 21.425`, zero overflow,
  `1.512s/step`, and offline W&B `jde950ee`, but feature loss drifts
  `0.655132 -> 0.703820`. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_checkpoint_media.jsonc`
  then tests a scheduled catch-up (`feature=1/probe=10` for global 800-900,
  then `feature=0.5/probe=20` for 900-1000). It is a useful nonpassing row:
  feature loss recovers `0.703862 -> 0.643852` at `1.308s/step`, but probe PSNR
  gives back a little (`21.428 -> 21.382`) and `pass=false` because probe loss
  does not decrease end-to-end. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.jsonc`
  then tests a constant Pareto objective (`feature=0.5`, `rgb_probe=40`) from
  the 1000-step state. It passes the combined gate with probe PSNR
  `21.384 -> 21.789`, zero overflow, and `1.461s/step`, but feature loss drifts
  `0.643823 -> 0.656728`. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.jsonc`
  then tests feature recovery from the 1100-step checkpoint: `feature=1/probe=20`
  for global 1100-1150, then `feature=0.75/probe=30` for 1150-1200. It is
  nonpassing but useful: feature loss recovers `0.656765 -> 0.635093` at
  `1.521s/step`, while probe PSNR gives back a little (`21.792 -> 21.738`) with
  zero overflow. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_checkpoint_media.jsonc`
  then tests a short probe-recovery continuation from the 1200-step checkpoint
  with `feature=0.75`, `rgb_probe=40` for 50 local steps. It passes with probe
  PSNR `21.740 -> 21.929`, zero overflow, and `1.523s/step`, but feature loss
  rises `0.635066 -> 0.638799`. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.jsonc`
  then tests a constant balance continuation from the 1250-step checkpoint with
  `feature=1.0`, `rgb_probe=40.0` for 50 local steps. It is the first current
  both-improving row: feature loss `0.638803 -> 0.632192`, probe PSNR
  `21.933 -> 21.963`, zero overflow, and `1.285s/step`. Config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.jsonc`
  extends the same objective from the 1300-step checkpoint for 100 local steps.
  It also passes and improves both metrics: feature loss `0.632124 -> 0.627129`,
  probe PSNR `21.965 -> 21.979`, zero overflow, and `1.690s/step`. The next
  timing-control config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_timing_repeat.jsonc`
  repeats the same 1300->1400 segment and reproduces the slower regime:
  `1.711s/step`, feature loss `0.632124 -> 0.627120`, probe PSNR
  `21.965 -> 21.979`, zero overflow, and `68/45/128` max/p95/cap tile count.
  The sparse-forward batched-VJP helper/media row preserves the same objective
  movement at mean step/backward/render `399.9/176.9/125.2ms` and last-20
  `262.9ms/step`, so the current speed problem is no longer the dense target
  VJP path. The effective-lr001 sparse-forward rerun keeps the dense lr001
  quality endpoint at mean step/backward/render `372.3/158.9/119.9ms`, feature
  loss `0.630549`, and probe PSNR `22.034`, but gives up lr005's better feature
  loss and has noisy late timing; the remaining gap is visual quality against
  the same-grid oracle.
  The whole-graph profile script
  `research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py`
  then loads the 1250 and 1300 checkpoints and manually splits the current
  target-grid/frozen-probe objective. It passes with zero overflow and reports
  renderer backward as `81.3-81.4%` of manual backward, but does not reproduce
  the trainer slowdown (`1565.9ms` manual total at global step 1250 versus
  `1504.0ms` at 1300). The trainer now records `step_timings_ms`, and two
  20-step no-media traces add the end-to-end view:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_trace20_from1250.jsonc`
  passes with `1705.3ms` mean step after dropping the first optimizer/warmup
  step, while
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_trace20_from1300.jsonc`
  exits cleanly but has `pass=false`: it is slower at `1850.7ms` no-first mean
  and spikes feature/probe loss at global step `1318`. The next quality/speed
  follow-up
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_chunktrace20_from1300.jsonc`
  traces chunks for global steps `1317/1318/1319`; it shows the spike is
  distributed (`27/32` chunks worsen), with `44.5%` of the weighted-loss delta
  in frames `0-15`, and the elevated loss persists at `1319`. The next
  optimizer/LR gate was run from the 1300-step checkpoint: after fixing the
  trainer to re-apply config LR after `optimizer.load_state_dict`, the retained
  optimizer row records loaded/effective LRs `[0.005] -> [0.001]` and passes
  with the spike removed (`d loss 1318-1317 = -0.000067`, end loss `0.884576`,
  feature loss `0.631648`, probe PSNR `21.991`, no-first `1384.4ms/step` /
  `748.9ms` backward). The reset-optimizer `lr=0.001` control also passes
  (`0.884902`, `0.631614`, `21.984`, no-first `1608.9ms/step` / `860.0ms`
	  backward). Continuing this exact quality schedule should use the 1300
	  checkpoint with effective `lr=0.001`; further speed work should beat the
	  sparse-forward batched-VJP helper, not chase tile overflow. The 100-step effective-lr001
  continuation from 1300 also passes with media/checkpoint: feature loss
  `0.632124 -> 0.630549`, probe PSNR `21.965 -> 22.034`, mean
  `1463.8ms/step`, `778.4ms` backward, zero overflow, and loaded/effective
  LRs `[0.005] -> [0.001]`. It avoids the early 1318 jump but later has a
  smaller transient at `1377->1378`; versus the older lr005 1300->1400 row it
  is faster and better on probe PSNR, but worse on final feature loss
  (`0.630549` vs `0.627129`) and slightly worse weighted loss (`0.880942` vs
  `0.880751`). The matched effective-lr001 sparse-forward rerun preserves that
  dense lr001 endpoint at `372.3ms/step` mean and `158.9ms` backward, but it
  keeps the same quality tradeoff and noisy late timing. The checkpoint-selection
  gate from global step 1400 then picks the lr005-sparse state: 50 effective-lr001
  steps from that checkpoint pass to feature loss `0.625976` and probe PSNR
  `22.010`, while the lr001-sparse state fails after a `1444 -> 1445` jump and
  ends at feature loss `0.631770` / probe PSNR `21.843`. The selected
  lr005-sparse 1450->1500 media gate then passes with loss
  `0.877762 -> 0.876224`, feature loss `0.625962 -> 0.625428`, probe PSNR
  `22.010 -> 22.027`, mean `315.8ms/step`, last-20 `254.0ms/step`, and valid
  but still blurry probe media. The first explicit full-resolution autograd
  RGB-aux probe-init bridge from that sparse 1500 checkpoint is a negative
  quality result: RGB loss improves `0.272626 -> 0.259968`, but feature loss
  worsens `0.625418 -> 0.626799`, frozen-probe PSNR drops
  `22.028 -> 21.879`, trainable-colorizer media artifacts appear, and mean
  step time is `5206.6ms` (`16.5x` slower than sparse 1500). The rendered-feature
  sparse-pixel RGB probe from the same sparse 1500 checkpoint then passes as a
  sampled-colorizer diagnostic (`0.168261 -> 0.099014`, sparse PSNR
  `7.740 -> 10.043`, `241.4ms/step`) but dense full-video PSNR is only
  `6.096` and media remains sparse-streaked, so distribution-matched colorizer
  training alone is not a quality bridge. The denser stratified64 follow-up
  samples `262,144` full-resolution pixels/step (`4x` the previous rendered
  probe) and still reaches only `6.132` dense full-video PSNR at
  `331.5ms/step`, so target-grid sampling bias is not the explanation. The
  first native sparse visual VJP gate updates STAR parameters from sparse RGB
  loss (`model_grad_seen=true`, frozen colorizer) at `336.8ms/step`, but
  full-video PSNR worsens to `5.739`; it proves the sparse visual-VJP bridge,
  not quality. The joint sparse visual VJP follow-up trains STAR and the
  hidden64 colorizer together (`model_grad_seen=true`,
  `colorizer_grad_seen=true`) and improves dense full-video PSNR to `6.025`,
  but it still trails the colorizer-only stratified diagnostic (`6.132`) and
  costs `729.4ms/step`. This proves joint gradients, not visual promotion. The
  mixed target-grid/probe plus sparse visual VJP gate is now done too: it
  preserves feature/probe movement and raises sparse visual sample PSNR to
  `6.036`, but dense full-video PSNR stays `6.024` while timing slows to
  `964.0ms/step`. The patch2x2 same-pixel support follow-up is faster
  (`619.5ms/step`) and raises sparse visual sample PSNR to `6.179`, but
  feature-target loss worsens and dense full-video PSNR drops to `6.000`. The
  patch-mean64 visual-basis follow-up samples `1,048,576` sparse visual
  pixels/step, pools them into `262,144` local-mean cells, restores
  feature/probe movement and dense full-video PSNR to `6.023`, but costs
  `1124.6ms/step` and still has sparse/high-frequency media. Target-area64
  keeps the same support and compares against true area-downsampled RGB target
  cells; it is slightly faster (`1103.1ms/step`) and raises sparse visual PSNR
  to `6.064`, but dense RGB/media are unchanged. Phased target-area64 cycles the
  same compact `2x2` support across a `4x4` subcell schedule; it raises sparse
  visual PSNR to `6.077`, but dense RGB falls to `6.019` at `1169.2ms/step`.
  Full-cell8 target-area support sends gradients through every dense pixel
  (`16,777,216` visual pixels/step) and is nonpassing: feature/probe losses
  worsen, dense RGB falls to `5.722`, and mean step is `7526.7ms` with
  `5702.6ms` in sparse visual loss construction. Manual hidden64 VJP matches
  the same endpoint while cutting sparse visual loss construction to
  `3803.6ms` and mean step to `6414.0ms`, so it is a parity scaffold rather
  than a promotion. Star-only manual hidden64 skips colorizer parameter
  gradients and cuts mean step further to `5801.7ms`, but dense RGB falls to
  `5.648`, so it is only a lower-bound diagnostic. Fast-GELU manual hidden64
  keeps colorizer gradients but is rejected: `6252.1ms/step`, same bad `5.722`
  dense RGB, and a worse profiled loss-side total than exact manual. The compact
  manual-linear variant is the first affordable full-cell8 mechanics gate: the
  linear probe reaches only `16.980` full-video PSNR, but the trainer row passes
  mechanically at `2064.4ms/step` with sparse visual loss construction down to
  `383.3ms`; it still is not a quality route because feature loss slightly
  worsens and dense RGB is only `5.668`. The hidden32 manual VJP follow-up keeps
  most hidden64 probe capacity (`19.704` full PSNR vs `20.073`) but remains too
  slow at `4298.4ms/step` with `2136.1ms` sparse visual loss construction and
  dense RGB only `5.678`, so shrinking the hidden decoder in Python is also not
  the route. The native target-area hidden64 follow-up is the first positive
  full-support native port: it cuts the matched star-only trainer row
  `5801.7 -> 3496.0ms/step`, and 512px native-only synthetic support passes where
  the Torch hidden-VJP baseline OOMs, but dense RGB stays `5.648`. The hidden32
  native follow-up cuts mean step further to `2464.6ms` and sparse visual
  backward to `1321.7ms`, but fails the gate with probe PSNR `19.481` and full
  RGB `5.632`. The split full-step profiles say target-area reduction is only
  `~0.12-0.13s`; exact GELU backward (`~1.34-1.44s`) and fc1 (`~0.75-0.89s`)
  dominate the old Python loss-side path. A benchmark-only skip-feature-grad
  mode shows raw feature-gradient atomics are also small in the native hidden64
  path: backward improves only `594.9 -> 562.2ms` at 256px and
  `1918.6 -> 1854.3ms` at 512px. The opposite feature-only split reaches the
  same conclusion: full/feature-only/geometry-only native backward is
  `581.3/548.2/547.3ms` at 256px and `1919.7/2106.7/2174.0ms` at 512px, so
  shared hidden64 recompute/traversal dominates simple output-gradient masking.
  Recompute-only disables all output-gradient atomics and still costs
  `571.3ms` backward at 256px and `2101.7ms` at 512px, so the shared
  replay/hidden64 VJP envelope is the floor. Traversal-only skips hidden64 VJP
  too and drops backward to `194.9ms` at 256px and `742.2ms` at 512px, putting
  the hidden64 forward/VJP slice at roughly `376.5ms` and `1359.6ms`.
  Hidden-forward-only splits that slice into forward `150.6/450.6ms` and
  backward `225.8/909.0ms` at 256/512px, making W^T/GELU feature VJP the
  larger hidden subtarget. Hidden-preact-only narrows that again: output+GELU
  prebackward is only `54.8/61.7ms`, while the F32 W^T feature-gradient matvec
  is `171.0/847.3ms`. The exact row-major W^T follow-up preserves
  full-gradient parity but is rejected: full native backward gets slower
  (`647.4 -> 711.5ms` at 256px and `2040.5 -> 2161.6ms` at 512px), while the
  recompute-only floor only improves slightly (`572.1 -> 555.8ms`,
  `1993.0 -> 1983.4ms`). The exact vec4 W^T follow-up is positive in the
  direct kernel: same-build full backward improves `675.9 -> 642.2ms` at 256px
  and `2408.1 -> 1804.7ms` at 512px, with a 512px repeat at `1832.8ms`; the
  recompute floor improves `586.6 -> 518.3ms` and `2305.2 -> 1635.8ms`.
  The current-build trainer A/B promotes `native_hidden64_target_area_star_only_vec4_wt`
  as the preferred full-support native target-area star-only mode: mean step
  improves `4262.1 -> 4071.0ms`, mean backward `3700.2 -> 3152.6ms`, and mean
  sparse visual backward `2546.7 -> 1963.5ms`, with matched endpoint class.
  The 50-step promoted-mode gate passes too (`3359.2ms` mean step,
  `3072.1ms` last step, `5.732` full RGB, zero overflow), but still trails the
  compact target-area64 helper route on the fresh current-build gate
  (`930.6ms`, `6.023` full RGB), so this is a full-support baseline, not the
  final objective. The compact native star-only diagnostic is rejected too:
  `2265.0ms` mean step, no colorizer gradients, and slower sparse visual
  backward than compact autograd. The compact manual-hidden64
  colorizer-gradient diagnostic is also rejected: it records
  colorizer grad required/seen `true/true`, but costs `2007.4ms` mean /
  `1899.2ms` no-first step, worsens feature loss `0.625418 -> 0.626795`,
  drops probe PSNR `22.028 -> 21.860`, and trails compact autograd's first-five
  timing (`991.9ms` mean / `787.7ms` no-first step). The native
  colorizer-gradient vec4 W^T implementation passes tiny parity for STAR and
  colorizer parameter gradients, but the compact trainer gate is a reject:
  `2738.7ms` mean step, `1474.2ms` backward, colorizer grads seen, zero
  overflow, and the same feature/probe regression as manual hidden64. The
  colorizer-gradient-only split then isolates the compact native failure:
  direct native backward at the same `64f/512px/8192t`, `6.25%` support is
  `88.9ms` for star-only vec4 W^T, `536.6ms` for colorizer-grad-only, and
  `531.4ms` for full colorizer vec4 W^T. The blocker is the naive per-pixel
  colorizer parameter-gradient atomic envelope, not STAR feature/geometry
  gradients. A Torch/MPS sidecar reducer prototype is correct
  (`1.26e-08` colorizer max error, `2.62e-10` STAR max error) and beats native
  atomics in the same-window direct gate (`390.9ms` vs `752.8ms`), but it still
  trails the sparse-pixel baseline (`276.6ms`) because it duplicates sparse
  render plus target-area hidden replay. The same-pass SIMD-reduce follow-up is
  the correct atomic-shape fix at the direct-kernel boundary: compact native
  colorizer total/backward becomes `297.2/239.2ms` in the matched run versus
  sparse-pixel `312.1/31.6ms`. The trainer still rejects it, though:
  `2908.9ms` mean step, `1363.0ms` backward, `604.0ms` sparse visual backward,
  and the same feature/probe regression. Keep compact autograd as the practical
  visual route; native work must remove whole-graph target-area overhead or
  change the objective/support, not just reduce colorizer atomics.
  The next visual gate should change objective/support or reduce hidden64 native
  reverse recompute, not rearrange sparse RGB samples or chase feature atomics
  alone. The first explicit
  optimizer-LR schedule gate (`0.001` until
  global step `1375`, then `0.00025`) passes mechanically but is negative for
  promotion: it removes the `1377->1378` jump, then a comparable jump appears
  at `1385->1386`; the scheduled 100-step row ends worse than static lr001 on
  weighted loss (`0.881602` vs `0.880942`), feature loss (`0.630803` vs
  `0.630549`), probe PSNR (`22.027` vs `22.034`), and timing (`1506.9ms` /
  `807.2ms` backward vs `1463.8ms` / `778.4ms`). A diagnostic 88-step late
  trace is expected to fail the quality pass bit because it stops at the spike;
  it attributes `1385->1386` to distributed degradation (`26/32` chunks
  worsen, summed weighted-loss delta `0.015248`, max frame `0` chunk
  `0.001802`). Checkpoint selection is resolved in favor of the lr005-sparse
  lineage; the next quality move should change the objective/model bridge, not
  lower LR forever. Generated reports:
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_continuation_chain.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_wholegraph_profile.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace_report.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace_report.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_reset_report.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_100step_report.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_schedule_report.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_checkpoint_select_1400_to1450.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_lr005sparse_1450_to1500_media.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_autograd_rgbaux_probeinit_from1500_negative.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels.md`,
  `outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_stratified64.md`,
  and
  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe.md`,
  and
  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe.md`,
  and
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_linear_from1500.md`.
- The STAR V-JEPA vs Gaussian/token comparison is now normalized in
  `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`.
  The matched 64f/512px/8192 rows are: STAR V-JEPA streaming target
  `3.743s/step`; STAR V-JEPA cached-chunks target `1.655s/step`; STAR V-JEPA
  target-grid loss `1.351s/step` (`1.451s/step` for the 20-step media row);
  STAR target-grid RGB-aux1/aux10 about `2.000s/step` for 20-step probes and
  `1.876s/step` for the 100-step aux10 row, with the negative RGB-warm20 row at
  `1.639s/step`; standalone target-grid feature-to-RGB oracle `0.00243s/step`;
  integrated frozen-probe rows `1.220s/step`, `1.268s/step`, `1.355s/step`,
  the checkpoint/no-media repeat at `1.268s/step`, the resumed
  300-step continuation at `1.440s/step`, and the probe-emphasis 600->800
  continuation at `1.512s/step`, plus the nonpassing scheduled 800->1000
  balance row at `1.308s/step` and the passing feature0.5/probe40 1000->1100
  continuation at `1.461s/step`, plus the nonpassing recover schedule row at
  `1.521s/step`, plus the passing feature0.75/probe40 1200->1250 row at
  `1.523s/step`, plus the passing feature1/probe40 1250->1300 row at
  `1.285s/step`, plus the passing feature1/probe40 1300->1400 dense extension
  at `1.690s/step`, its timing repeat at `1.711s/step`, the lr005
  sparse-forward batched-VJP helper/media row at `0.400s/step` mean /
	  `0.263s/step` last-20, and the lr001 sparse-forward rerun at `0.372s/step`
	  mean / `0.539s/step` last-20, plus the selected lr005-sparse 1450->1500
	  media gate at `0.316s/step` mean / `0.254s/step` last-20, plus the negative
	  autograd RGB-aux probe-init bridge at `5.207s/step`, plus the
	  rendered-feature sparse-pixel RGB probe at `0.241s/step` and the
		  stratified64 rendered-pixel probe at `0.332s/step`, plus the sparse visual
		  VJP frozen-probe gate at `0.337s/step`, the joint sparse visual VJP gate
		  at `0.729s/step`, and the mixed target-grid/probe+sparse visual VJP gate
		  at `0.964s/step`, the patch2x2 support gate at `0.620s/step`, and the
		  patch-mean64 visual-basis gate at `1.125s/step`, plus the target-area64
		  visual-basis gate at `1.103s/step`, plus the phased target-area64
		  visual-basis gate at `1.169s/step`, plus the full-cell8 target-area
		  gate at `7.527s/step` and its manual hidden64 VJP variant at
		  `6.414s/step` plus star-only manual hidden64 at `5.802s/step`
			  and fast-GELU manual hidden64 at `6.252s/step`, plus compact manual-linear
			  full-cell8 at `2.064s/step`, manual hidden32 at `4.298s/step`, and
			  native full-cell target-area star-only at `3.496s/step`,
		  plus the matched 512px native handoff gate where `fused_first3`
		  totals `1.153s` and `logit_handoff_reduce_vec4` has `0.386s`
		  native backward plus `0.422s` prep;
  STAR RGB fast diagnostic
  `2.491s/step`; Gaussian/token recon-only cached conditioning
  `3.460s/step`, `1.963s` backward; and Gaussian/token
  prediction-side V-JEPA loss `38.621s/step`, `36.762s` backward. The multicam
  cached-V-JEPA rows
  remain useful 16f/128px references, not matched 64f/512px evidence. The
  refreshed all-renderer matrix is
  `outputs/benchmarks/2026-05-19_renderer_scaling_report.md`.
- `gradcache_cached_bins` / `feature_direct_gradcache_cached_bins` reuses the
  forward tile bins in backward. It passes parity and gives a same-session
  synthetic renderer-backward win (`1068.0ms -> 935.8ms`), but the first-class
  512px/8192t/chunk2 trainer row does not improve end-to-end and is slower on
  measured backward than plain gradcache in the same session. Keep it as a
  sidecar diagnostic, not the default.
- `gradcache_feature_grad_only` is a benchmark-only diagnostic that keeps only
  feature-gradient atomics. `gradcache_two_pass_feature_grad` composes
  `gradcache_skip_feature_grad` for geometry/opacity with
  `gradcache_feature_grad_only` for features. Tiny F4/F32 parity passes, but
  naive two-kernel split-recompute is not the next implementation path.
- `direct_feature_mode_matrix.py` now covers cached-bin modes and 512px. The
  39-row 64f/32768t/F32 matrix passes at 128/256/512px and writes
  `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_mode_matrix_128_256_512_64f_32768t_f32_cached/summary.md`.
  At 512px, `gradcache_cached_bins` is the fastest full-gradient total row
  (`1.979s`, `1.103s` backward), but the benchmark-only
  `gradcache_skip_feature_grad` diagnostic is still faster (`1.714s`, `0.804s`
  backward), so feature-gradient accumulation remains the next shader target.
- The refreshed feature-gradient-only/two-pass matrix lives at
  `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_two_pass_feature_grad_matrix_256_512_64f_32768t_f32/summary.md`.
  All 8 rows pass after the harness now uses an artifact-local `TMPDIR` and
  deletes stale case JSON before each subprocess. At 256px, two-pass is
  `1.343s` total / `1.063s` backward versus full gradcache `0.972s` /
  `0.692s`; at 512px, two-pass is `2.471s` / `1.613s` versus `2.467s` /
  `1.379s`. The reverse-order 512px rerun also stays negative, so the next
  "two-pass" candidate must avoid duplicate traversal with true fixedbin or
  tile-slot accumulation.
- `tile_slot_accumulator_budget.py` now records the fixedbin/tile-slot
  feasibility budget:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_tile_slot_budget_128_256_512_64f_32768t_f32/summary.md`.
  At 64f/32768t/F32, one feature-gradient atomic per tile slot and channel
  would reduce writes by `128x`, but naive prefix recompute is `64.4x` at
  128px, `39.8x` at 256px, and `10.8x` at 512px. A scalar f32 weight tape is
  `0.499/1.171/1.195GiB` at 128/256/512px; a per-channel tape is `16.0/37.5/38.2GiB`.
  This points the next shader at compact scalar prefix/weight storage or native
  image-space VJP, not per-channel tape storage.
- The reducer-only tile-slot isolation matrix lives at
  `outputs/benchmarks/2026-05-19_star_uvt_feature_tile_slot_reduce_matrix_256_512_64f_32768t_f32/summary.md`.
  All 16 rows pass. `gradcache_feature_grad_only_reduce_vec4` improves
  feature-only backward by `0.844x` at 256px (`532.8 -> 449.9ms`) and `0.891x`
  at 512px (`869.1 -> 774.8ms`). The full-gradient refresh at
  `outputs/benchmarks/2026-05-19_star_uvt_feature_full_reduce_refresh_256_512_64f_32768t_f32/summary.md`
  shows `gradcache_reduce_feature_grad_vec4` is a small 256px win
  (`676.6 -> 654.5ms` backward) and a larger 512px synthetic win
  (`1284.2 -> 1108.0ms`). Two-pass reduce composition is still not promoted:
  it improves the 256px two-pass row but remains slower than single-pass
  gradcache, and at 512px it loses/ties because duplicate traversal dominates.
- `fused_first3_sigmoid_mse` is a benchmark-only RGB handoff prototype. It
  computes a narrow `alpha * sigmoid(feature[:3]) -> mean MSE` VJP inside the
  STAR feature backward kernel. It is not the learned F32 `FeatureToColor` path,
  but it validates the handoff direction without the barrier-heavy reduction.
  The matched `64f/512px/8192t/F32` rerun keeps this direction alive:
  `494.09ms` backward / `1152.58ms` total with parity and zero overflow.
- `direct_linear_sigmoid_mse_backward` is the generalized benchmark handoff for
  a real `[3,F]` linear colorizer, bias, sigmoid, MSE target, and colorizer
  parameter gradients. It passes F4/F32 parity, but its target timing is slower
  than gradcache (`615-619ms` backward versus `477.5ms` same-session
  gradcache). The skip-colorizer-gradient variant was noisy and did not prove
  colorizer parameter atomics are the main bottleneck.
- `direct_logit_handoff_backward` is the image-space-prep handoff: image-space
  computes `grad_logits` and `grad_alpha`, while Metal applies
  `W^T @ grad_logits` in the STAR reverse traversal. It passes F4/F32 parity
  but is still slower than gradcache on the target row. The follow-up
  `logit_handoff_reduce` / `logit_handoff_reduce_vec4` gate combines that
  handoff with stable-tile feature-gradient reducers. All 256/512 direct rows
  pass parity and zero overflow; vec4 improves synthetic backward
  `571.7 -> 510.6ms` at 256px and `654.8 -> 642.3ms` at 512px, while scalar
  reduce regresses 512px backward to `722.5ms`. Keep vec4 as a diagnostic
  candidate only until a first-class trainer row proves the end-to-end win. The
  matched 512px/8192t rerun shows the same boundary more clearly:
  `logit_handoff_reduce_vec4` has `386.26ms` native backward but `421.89ms`
  Torch prep, so the next shader should fuse prep or avoid dense prep. The
  native-prep follow-up does that for the linear sigmoid-MSE handoff:
  `logit_handoff_reduce_vec4_native_prep` passes F4/F32 parity, drops matched
  512px prep `413.64 -> 37.29ms`, drops prep+backward `826.35 -> 428.98ms`,
  and drops total `1446.53 -> 1108.50ms`; it is still benchmark-only and does
  not cover hidden frozen-probe V-JEPA trainer loss. The hidden sigmoid-MSE
  native follow-up covers the hidden RGB/loss shape and passes parity, but it is
  not the speed answer: H32 scalar totals `317.54/610.90/2549.39ms` at
  128/256/512px, H64 256px totals `817.27ms`, and vec4 reduce is slower than
  scalar. The sparse hidden cached-bin follow-up is the first positive native
  port of that lesson: it reuses sparse bins, fuses hidden RGB/loss VJP only over
  selected pixels, and at 64f/512px/8192t/F32 drops H32 sparse64 total
  `29.66 -> 18.47ms`, H32 sparse128 `111.17 -> 64.17ms`, and H64 sparse64
  `45.09 -> 28.40ms`, all with parity and zero overflow. It is still a
  sparse visual boundary, not dense full-frame parity. The first trainer-wired
	  pixel64 native gate matches the manual endpoint (`3.26e-08` final sparse-loss
	  diff) but is neutral on speed: warm sparse loss+backward is `113.25ms` manual
	  versus `116.27ms` native, and warm step time is `405.97ms` versus `403.83ms`.
		  The target-area/full-support native port now passes as the next compact
			  visual-gradient gate: matched star-only trainer time drops
			  `5801.7 -> 3496.0ms/step`, but the dense RGB endpoint stays `5.648`, so it is
			  speed/memory evidence rather than visual-quality promotion. The hidden32
			  native target-area follow-up is faster (`2464.6ms/step`) but rejected
			  (`pass=false`, probe PSNR `19.481`), so shrinking decoder capacity is not
			  the recompute fix. The skip-feature-grad diagnostic cuts hidden64 native
			  target-area backward by only `3-6%`, so raw feature atomics are not the
			  decisive bottleneck. A
  first real-video linear RGB-VJP profile now passes from the 64f/512px/8192t
  1300-step checkpoint: manual `logit_handoff_reduce_vec4` matches autograd
  model/colorizer gradients (`9.43e-09` max abs, zero loss error) and measures
  `1691.0 -> 1587.4ms` (`1.065x`) with zero overflow. This is compatibility
  evidence for linear RGB loss, not for target-grid V-JEPA MSE or the hidden64
  frozen-probe objective. The target-grid/frozen-probe VJP bridge profile now
  covers that current keeper objective directly: it matches normal autograd at
  `2.57e-08` max gradient error with zero loss error and zero overflow, but
  the first autograd-image bridge is a slight negative (`1545.5ms` autograd
  versus `1594.3ms` bridge). The analytic target-grid/probe VJP follow-up keeps
  parity (`3.07e-08` max gradient error) and gives a small repeat-5 win
  (`1510.6 -> 1477.2ms`, `1.023x`). Treat this as evidence for an
  analytic/native target-grid/probe VJP trainer gate. That trainer gate now
  exists as `feature_target.image_vjp_mode=analytic` and passes the matched
  5-step 64f/512 smoke, but it is an end-to-end tie rather than a promotion:
  autograd mean step `1303.6ms`, warm analytic rerun `1304.6ms`, no-first
  `1264.1ms` versus `1259.2ms`, with `103.3ms` less backward offset by higher
  loss/VJP time. The sparse-pixel follow-up is the first version that wins the
  current keeper trainer loop: `feature_target.image_vjp_mode=
  analytic_sparse_pixels` packs only nonzero target-grid image-gradient pixels
  and calls a sparse direct-atomic Metal backward using forward bins. The
  repeat-3 profile passes parity (`4.61e-08` max grad error, zero loss error),
  visits `65,536` sparse pixels per 64f/512 step (`0.390625%` of dense), and
  cuts dense analytic bridge total `1245.9ms -> 920.5ms`; sparse renderer
  backward is `46.3ms` versus dense `557.6ms`, with `184.0ms` still spent
  packing from the dense Torch VJP. The matched 5-step trainer smoke passes and
  cuts no-first step `1318.0ms -> 973.7ms` while matching dense loss/probe
  PSNR. The direct sparse-grid follow-up now eliminates that dense VJP/packing
  step for the current trilinear target-grid shape:
  `feature_target.image_vjp_mode=analytic_sparse_grid` maps target-grid/probe
  gradients directly to sparse source pixel ids/values, keeps profile parity
  (`4.60e-08` max grad error), cuts bridge total to `760.6ms`, and passes the
  matched 5-step trainer smoke at `795.3ms` no-first step and `88.6ms`
  no-first backward. The sparse-grid render-mode matrix keeps
  `feature_direct_gradcache_reduce_vec4` selected under the new VJP path:
  no-first `730.5ms`, mean backward `78.3ms`, zero overflow, with gradcache at
  `759.4ms` and direct atomic at `779.3ms`. The sparse-forward follow-up then
  removes the remaining dense feature-image render for this objective:
  `feature_target.image_vjp_mode=analytic_sparse_grid_forward` renders only the
  same `65,536` support pixels, matches dense feature/alpha values exactly,
  and initially cut forward render `515.9ms -> 70.5ms` (`7.322x`) with a
  `492.3ms` no-first trainer row. The follow-up 128/256/512 scale matrix passes
  all rows with zero overflow, but exposes timing instability: sequential
  no-first step is `379.2ms` at 128px, `494.2ms` at 256px, and `973.0ms` at
  512px; an isolated 512px repeat immediately after the matrix lands at
  `598.2ms` no-first / `477.6ms` last step and `172.5ms` no-first backward.
  A dedicated 512px repeat-3 timing gate passes all rows with zero overflow and
  gives no-first step mean/min/max/stdev `504.9/411.0/626.4/110.3ms`, last-step
  `468.8/409.3/549.9/72.7ms`, and no-first backward
  `142.2/114.7/174.4/30.1ms`. Keep sparse-forward plus sparse-grid VJP plus
  vec4 reduce as the selected target-grid/frozen-probe diagnostic, but report
  timing with repeat context.
  The batched target-grid/probe VJP path is now the selected opt-in trainer
  mode for this diagnostic. The preflight stacks all 32 chunks after sparse
  forward, preserves loss and sparse gradient packs (`7.45e-09` loss error,
  `6.55e-11` max feature grad error), and cuts isolated target/probe loss+VJP
  `38.0ms -> 4.8ms` (`7.99x`). The first-class trainer mode
  `analytic_sparse_grid_forward_batched` passes the same 5-step checkpoint gate
  with zero overflow and repeat-3 no-first step mean/min/max/stdev
  `179.3/159.7/215.6/31.5ms`, no-first backward
  `72.0/60.8/90.2/15.9ms`, and no-first render `71.1/67.8/77.4/5.5ms`.
  The 100-step media/helper gate also passes:
  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_64f512_from1300_100step_media.md`
  records loss `0.886537 -> 0.880744`, feature loss
  `0.632124 -> 0.627122`, probe PSNR `21.965 -> 21.979`, zero overflow,
  mean step/backward/render `399.9/176.9/125.2ms`, and last-20
  `262.9/109.4/94.0ms`. It writes valid RGB-probe media and a 1400-step
  checkpoint, but the contact sheet remains blurry; native target-grid/probe
  loss+VJP or scalar fixedbin/tile-slot kernels must beat this repeat/100-step
  timing surface, while the next training gate is visual quality.
- The selected visual-quality gate is now explicit and failing:
  `outputs/benchmarks/2026-05-20_star_uvt_selected_visual_quality_gate.md`.
  The compact target-area route passes mechanics and speed (`930.6ms` mean
  step, zero overflow), but dense full RGB is only `6.023` PSNR, sparse visual
  PSNR is `6.064`, and the contact/probe sheets remain sparse-streaked or
  blurry. Do not scale this route to 300 videos until the objective/model bridge
  changes and dense media improves toward the RGB STAR bracket (`12.444` PSNR).
- The first trainable low-frequency RGB-grid bridge is also a negative
  visual-quality gate:
  `outputs/benchmarks/2026-05-20_star_uvt_rgb_grid_lowfreq_bridge_gate.md`.
  `feature_target.rgb_grid_loss_weight=40` trains the actual output colorizer
  through the fast target-grid sparse VJP path and passes mechanics at
  `353.1ms` mean step / `289.9ms` no-first, but dense full RGB drops to
  `5.657` PSNR while feature loss worsens `0.625418 -> 0.630230`. Low-frequency
  grid RGB is now a cheap diagnostic, not a scale-up route.
- The combined compact target-area plus RGB-grid bridge is also rejected:
  `outputs/benchmarks/2026-05-20_star_uvt_compact_rgbgrid40_visual_bridge_gate.md`.
  It passes mechanics with zero overflow and improves RGB-grid, frozen-probe,
  and sparse visual PSNR, but slows to `1647.9ms` mean step, worsens feature
  loss `0.625418 -> 0.630296`, and reaches only `5.720` dense full RGB PSNR.
  This rules out "low-frequency grid stabilization plus compact support" as
  the missing visual fix.
- The dense alpha/coverage diagnostic now explains the failure mode:
  `outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_failure_diagnostic.md`.
  Across compact, RGB-grid, and compact+RGB-grid checkpoints, normal PSNR is
  only `5.657-6.023`, but forcing alpha to one raises PSNR to
  `11.450-14.616` and target-background oracle composition reaches
  `20.149-25.562`. Alpha `>0.1` covers only `41.5-43.5%` of pixels, so the
  next visual gate should directly address visibility/coverage/composition.
- The direct sampled alpha-to-one follow-up is also rejected:
  `outputs/benchmarks/2026-05-20_star_uvt_compact_alpha1_coverage_gate.md` and
  `outputs/benchmarks/2026-05-20_star_uvt_alpha1_dense_alpha_diagnostic.md`.
  It adds `sparse_visual.alpha_loss_weight=1.0` on the same compact
  target-area support. The sampled alpha loss improves `0.752440 -> 0.738210`
  and sparse visual PSNR improves `5.678 -> 6.061`, but pass is `false`,
  feature loss worsens `0.625418 -> 0.627071`, RGB-probe PSNR drops
  `22.028 -> 21.900`, dense full RGB stays `6.018`, and dense alpha `>0.1`
  is unchanged at `43.1%`. Same-support alpha pressure is not enough.
- The phase-covered sampled alpha-to-one follow-up is rejected too:
  `outputs/benchmarks/2026-05-20_star_uvt_phase_alpha1_coverage_gate.md` and
  `outputs/benchmarks/2026-05-20_star_uvt_phase_alpha1_dense_alpha_diagnostic.md`.
  It cycles `patch_phase_shape=[4,4]` with `2x2` compact support and keeps zero
  overflow. Sampled alpha loss improves `0.751768 -> 0.739891` and sparse
  visual PSNR improves `5.694 -> 6.072`, but feature loss worsens
  `0.625418 -> 0.626961`, frozen-probe PSNR drops `22.028 -> 21.904`, dense
  RGB falls to `6.014`, and dense alpha `>0.1` falls to `43.0%`. Sparse support
  phase cycling is not the missing coverage bridge.
- The target-aware black-hole coverage follow-up is rejected too:
  `outputs/benchmarks/2026-05-20_star_uvt_blackhole4_coverage_gate.md` and
  `outputs/benchmarks/2026-05-20_star_uvt_blackhole4_dense_alpha_diagnostic.md`.
  It adds `sparse_visual.black_hole_loss_weight=4.0` on compact target-area64
  support. The black-hole loss improves `0.262537 -> 0.256889` and sparse
  visual PSNR improves `5.678 -> 6.059`, but feature loss worsens `0.625418 ->
  0.627272`, frozen-probe PSNR drops `22.028 -> 21.890`, dense RGB stays
  `6.014`, and dense alpha `>0.1` stays `43.0%`. Target-aware same-support
  empty-pixel pressure is not the missing dense coverage bridge.
- Target-background composition is informative but rejected as a visual route:
  `outputs/benchmarks/2026-05-20_star_uvt_target_background_composition_gate.md`
  and
  `outputs/benchmarks/2026-05-20_star_uvt_targetbg_alpha1_dense_alpha_diagnostic.md`.
  `sparse_visual.composition=target_background` mechanically passes and
  improves feature/probe plus sparse visual PSNR, but dense full RGB is only
  `5.666`. Adding `alpha_loss_weight=1.0` restores alpha coverage to `43.1%`
  but regresses feature/probe and reaches only `5.748` dense RGB. Forced-alpha
  PSNR is much better (`14.891-14.899`) and target-background oracle reaches
  `27.105-27.443`, so the color/content path can improve, but the rendered
  black-background video is still coverage-limited.
- The alpha-sweep and patch4 support follow-up is also rejected:
  `outputs/benchmarks/2026-05-20_star_uvt_patch4_support_alpha_sweep_gate.md`,
  `outputs/benchmarks/2026-05-20_star_uvt_alpha_sweep_dense_diagnostic.md`, and
  `outputs/benchmarks/2026-05-20_star_uvt_patch4_alpha_sweep_dense_diagnostic.md`.
  The alpha sweep shows `16x` posthoc alpha gain only reaches `8.337-8.592`
  PSNR on target-background checkpoints, while alpha floors recover the
  forced-alpha result. The `4x4` support pilot raises sparse visual support to
  `25%` of dense pixels and improves sparse visual PSNR `26.319 -> 27.251`,
  but total loss fails `1.631071 -> 1.637982`, feature loss worsens
  `0.625418 -> 0.626858`, frozen-probe PSNR drops `22.028 -> 21.878`, and
  dense RGB is only `5.698`. Denser sampled support is not the missing bridge.
- The raw-opacity bias follow-up is rejected:
  `outputs/benchmarks/2026-05-20_star_uvt_raw_opacity_bias_gate.md` and
  `outputs/benchmarks/2026-05-20_star_uvt_raw_opacity_bias_dense_diagnostic.md`.
  It rerenders compact, targetbg_alpha1, and patch4_targetbg_alpha1 after adding
  logit-space biases `[-2,-1,0,1,2,3,4]` to tube opacity. Bias `+4` is best for
  all three, but only reaches `6.194/5.926/5.871` PSNR and barely changes alpha
  `>0.1` coverage (`43.5 -> 46.5%`, `43.1 -> 45.8%`, `41.5 -> 44.2%`). Plain
  opacity scheduling is not the visibility bridge.
- The dense alpha-only support follow-up is rejected:
  `outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_support_gate.md` and
  `outputs/benchmarks/2026-05-20_star_uvt_densealpha075_dense_diagnostic.md`.
  The new opt-in `dense_alpha` trainer path sends dense `grad_alpha` through
  `gradcache_skip_feature_grad` while leaving feature/probe on the selected
  sparse-forward batched target-grid route. The 5-step `alpha_target=0.75`
  pilot writes artifacts but fails strict loss decrease: weighted loss
  `1.271702 -> 1.284505`, dense alpha loss `0.395507 -> 0.397107`, feature loss
  `0.625418 -> 0.626814`, RGB-probe PSNR `22.028 -> 21.861`, and dense RGB
  `5.647`, with dense-alpha render/loss/backward `834.45/124.58/858.91ms`.
  The diagnostic raises forced-alpha/oracle potential to `14.556/25.809` PSNR
  but lowers alpha `>0.1` to `40.7%`; dense alpha is a diagnostic gradient
  source, not the support bridge by itself.
- The alpha-only visibility speed profile is a positive diagnostic-only
  implementation detail:
  `outputs/benchmarks/2026-05-20_star_uvt_alpha_only_visibility_profile_full.md`.
  The new `render_uvt_feature_alpha_all_pixels_with_bins` wrapper uses the
  existing sparse-pixel Metal path with a dummy F1 feature and cached bins, so
  it avoids dense `[T,H,W,F32]` feature images for alpha-only work. On the same
  64f/512px/8192t dense-alpha checkpoint and all 32 frame chunks it matches
  dense alpha exactly, matches dense cached geometry/opacity gradients within
  `4.7e-7`, keeps zero overflow (`68/53/128` max/p95/cap), and cuts
  render+backward `1100.8 -> 634.6ms`. Use it for future alpha-only
  diagnostics, not as a scale or visual-quality promotion.
- The sparse-F1 dense-alpha trainer gate wires that wrapper behind
  `dense_alpha.render_mode=sparse_f1`:
  `outputs/benchmarks/2026-05-20_star_uvt_dense_alpha_sparsef1_trainer_gate.md`.
  It preserves the dense-F32 endpoint and remains pass-false: weighted loss
  `1.271702 -> 1.284505`, dense alpha `0.395507 -> 0.397107`, feature
  `0.625418 -> 0.626814`, RGB-probe PSNR `22.028 -> 21.861`, and dense RGB
  `5.647`. The useful result is speed only: mean step/backward
  `2558.6/1114.2 -> 873.3/370.0ms`, and dense-alpha render/loss/backward
  `834.5/124.6/858.9 -> 276.0/22.0/303.7ms`. This removes dense F32 alpha
  rendering as the excuse; the remaining blocker is support/objective quality.
- The CPU visibility support bridge prototype closes the next missing
  implementation-detail gate:
  `outputs/benchmarks/2026-05-20_star_uvt_visibility_support_bridge_cpu_proxy.md`.
  From a zero-hit target start, same-support dense alpha keeps target alpha
  `>0.10` coverage at `0.0`; a soft projected-tube coverage proxy sends
  center/velocity gradients, lowers proxy loss `45.109 -> 0.296`, raises target
  alpha mean `0.0 -> 0.092`, and raises target alpha `>0.10` coverage to
  `0.324`. This is a support-changing geometry-gradient proof only. It should
  be ported into the trainer before any 300-video scale run, but it is not a
  dense visual-quality or Metal-speed promotion.
- The first-class visibility-proxy trainer gate closes that trainer-port gap:
  `outputs/benchmarks/2026-05-20_star_uvt_visibility_proxy_trainer_gate.md`.
  From the sparse 1500 checkpoint, weighted loss improves
  `0.871986 -> 0.871864`, feature target loss `0.625418 -> 0.625379`,
  RGB-probe PSNR `22.0277 -> 22.0291`, and visibility-proxy loss
  `-4.20957 -> -4.20992` with center/velocity gradients seen and 4096 target
  points. Mean timing is `541.1ms` step, `306.6ms` backward, and `237.0ms`
  visibility-proxy work. This is a trainer mechanics pass, not a visual
  promotion: dense full RGB PSNR is still `5.640`, so the next gate should
  measure dense support/alpha movement or reduce the proxy cost before scale-up.
- The dense-support follow-up rejects the current center-only proxy as the
  quality bridge:
  `outputs/benchmarks/2026-05-20_star_uvt_visibility_proxy_support_gate.md`.
  Compared with the selected sparse step-1500 checkpoint, the 5-step proxy
  improves forced-alpha PSNR `11.722 -> 14.552` and target-background oracle
  `20.140 -> 25.834`, but alpha `>0.1` falls `41.1% -> 40.5%`. A stronger
  10x/20-step run improves proxy loss `-4.20957 -> -4.21215`, but fails trainer
  loss `0.834100 -> 0.844115`, worsens feature/probe losses, and still only
  reaches `40.6%` alpha `>0.1`. Do not scale this proxy to 300 videos without
  an explicit opacity/support term or support-density change.
- The opacity/precision support-aware proxy closes that missing implementation
  detail but is still rejected as the next scale bridge:
  `outputs/benchmarks/2026-05-20_star_uvt_visibility_support_proxy_gate.md`.
  The trainer now has `center_weight`, `support_weight`, and
  `support_epsilon`; the support term evaluates differentiable target-point
  coverage from opacity and precision, and focused tests prove opacity and
  precision gradients. The 5-step run passes mechanically with weighted loss
  `0.910498 -> 0.909964`, support proxy loss `3.4303 -> 3.3821`, RGB-probe
  PSNR `22.0277 -> 22.0289`, and gradients seen on raw opacity/precision, but
  feature loss slightly worsens `0.625418 -> 0.625436`. It is also too
  expensive as written: mean step/backward/proxy are
  `1186.8/841.8/693.7ms`. Dense support barely changes versus center-only:
  normal PSNR `5.640 -> 5.643`, forced-alpha `14.552 -> 14.553`, oracle
  `25.834 -> 25.820`, and alpha `>0.1` `0.405 -> 0.406`. Do not scale this
  objective to 300 videos; the next bridge needs cheaper/fused support density,
  opacity/support parameterization, or support birth/split.
- The fixed-budget support birth/split CPU gate is the first positive
  mechanism after the support-proxy rejections:
  `outputs/benchmarks/2026-05-20_star_uvt_visibility_birth_split_cpu_gate.md`.
  It starts from the same zero-hit support fixture family, keeps a fixed budget
  of `16` tubes, and reallocates `8` dead/miss tubes onto a fitted target
  support trajectory. Same-support alpha still cannot create coverage
  (`0.0000` target alpha `>0.10`), and the center support proxy reaches only
  `0.5784`; fixed-budget birth/split reaches `1.0000` target alpha `>0.10`
  immediately, then dense-alpha refinement preserves `1.0000` while reducing
  background alpha `0.0479 -> 0.0072` and loss `0.0233 -> 0.0033`. This is not
  a trainer or Metal promotion, but it points the next implementation step at
  a first-class dead/low-contribution tube reallocation path, not another
  sampled all-tube support proxy.
- The first-class support birth/split trainer gate now closes that trainer-port
  gap:
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_trainer_gate.md`.
  The trainer has `support_birth_split.enabled`, validates stale optimizer
  resume away, samples target points, fits a screen-space trajectory, and
  reallocates a fixed number of existing tubes before training while preserving
  the tube budget. From the sparse 1500 checkpoint, the 512px/64f gate
  reallocates `32/8192` low-opacity tubes, raises selected opacity
  `0.3418 -> 0.8000`, keeps zero overflow (`100/71/128` max/p95/cap), and
  passes 5 steps with weighted loss `0.910290 -> 0.909536`, feature target
  `0.635579 -> 0.635530`, RGB-probe loss `0.006868 -> 0.006850`, mean
  step/backward/render `189.4/55.6/70.1ms`, and full RGB PSNR `5.708`.
  This is real implementation progress and much cheaper than sampled all-tube
  support proxy work, but it is not visual-quality promotion. Next gate:
  birth/split dense-support diagnostics and a conservative
  `reallocate_tubes`/radius sweep, ideally using uncovered/low-alpha target
  pixels instead of top-brightness samples.
- The birth/split dense-support diagnostic keeps it in the "promising
  primitive" bucket:
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_dense_support_diagnostic.md`.
  Against start1500, center5, and support5, birth32 has the best
  black-background PSNR (`5.708`) and forced-alpha PSNR (`14.606`) and raises
  high-alpha coverage (`alpha>0.5` `0.117` versus `0.099` for center/support).
  It does not solve coverage: alpha `>0.1` is `0.411`, only back to start1500,
  and target-background oracle drops to `25.234` versus center/support
  `25.834/25.820`. Next experiment should sweep birth/split amount/radius and
  sample uncovered/low-alpha pixels instead of simply top-brightness pixels.
- The uncovered-brightness target sampler follow-up is now run:
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_uncovered_trainer_gate.md`
  and
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit32_uncovered_dense_support_diagnostic.md`.
  It adds `support_birth_split.target_point_source=uncovered_brightness`, uses
  a sampled sparse-F1 alpha pre-pass, and selects genuinely low-alpha bright
  targets (`selected_alpha_mean=0.0209`, `5243` candidates, `2048` selected)
  before reallocating `32/8192` low-opacity tubes. The 512px gate passes with
  zero overflow (`100/71/128`), mean step/backward/render
  `187.4/61.8/65.3ms`, weighted loss `0.900186 -> 0.899545`, feature target
  `0.634780 -> 0.634690`, RGB-probe loss `0.006635 -> 0.006621`, and dense
  full RGB PSNR `5.713`. This is mechanically useful but not a quality fix:
  dense support still has alpha `>0.1` `0.411`, forced-alpha PSNR `14.579`,
  and oracle `25.319`. Next gate is a conservative
  `target_point_source={uncovered_brightness,low_alpha}` plus
  `reallocate_tubes`/radius sweep, promoted only if dense alpha coverage rises
  without overflow.
- The first birth/split sweep gate is now recorded:
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_8row.md`,
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_8row_cap256.md`,
  and
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_n32_radius_cap128.md`.
  The harness
  `research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py`
  generates matched configs, runs the trainer, sets `STAR_UVT_TILE_CAPACITY`
  to match each row, and runs dense support diagnostics. Cap `128` rejects
  `64/128` births by overflow (`64`: max/p95/cap `132/103/128`, `12` overflow
  tiles; `128`: `196/167/128`, `16384` overflow tiles). Cap `256` makes all
  `64/128` rows valid, with best coverage at `low_alpha_n128_r96_cap256`:
  alpha `>0.1` `0.422`, normal PSNR `5.878`, forced-alpha PSNR `14.603`,
  oracle `23.623`, max tile `196/256`, mean step `212.8ms`. The safer cap-128
  corner says the same thing without higher tile capacity: `low_alpha_n32_r96`
  reaches alpha `>0.1` `0.420`, normal PSNR `5.825`, forced-alpha PSNR
  `14.591`, oracle `24.226`, max tile `100/128`, mean step `210.2ms`.
  Radius `32` rows are negative for coverage (`0.406-0.407`) despite better
  oracle (`25.704-25.716`). The read: wide support radius moves alpha coverage
  modestly, but oracle/content drops; do not launch 50 steps until an
  intermediate radius recovers oracle while keeping the coverage bump.
- The intermediate-radius follow-up answers that question negatively:
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_intermediate_radius_cap128.md`
  and
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep_intermediate_radius_cap128_dense_support.md`.
  With `32` births and cap `128`, uncovered-brightness passes every radius and
  gives a smooth tradeoff: `r64/r72/r80/r88` alpha `>0.1`
  `0.411/0.413/0.415/0.417`, alpha `>0.5`
  `0.119/0.124/0.129/0.136`, normal PSNR
  `5.713/5.734/5.757/5.782`, forced-alpha PSNR
  `14.579/14.587/14.592/14.596`, and oracle
  `25.319/25.187/25.015/24.802`. Low-alpha `r64/r72` passes with similar
  coverage but lower oracle, while low-alpha `r80/r88` fails loss decrease
  despite zero overflow (`r80` weighted/feature/probe worsen
  `0.913757->0.923922`, `0.638905->0.640059`, `0.006871->0.007097`; `r88`
  `0.925082->0.934971`, `0.642224->0.643357`, `0.007071->0.007290`). The read:
  radius alone is a blunt coverage/oracle tradeoff, so the next gate should
  change born-tube initialization such as opacity or anisotropic support before
  any long continuation.
- The born-opacity initialization sweep is now run:
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_opacity_init_sweep_r80_cap128.md`
  and
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_opacity_init_sweep_r88_cap128.md`.
  The sweep harness now supports `--opacities` and writes
  `support_birth_split.opacity` into generated configs. At `r80`, uncovered
  opacity `0.4/0.6/0.8/0.9` passes and moves alpha `>0.1`
  `0.414/0.414/0.415/0.415`, normal PSNR
  `5.735/5.748/5.757/5.760`, and oracle
  `25.177/25.083/25.015/24.987`; low-alpha only passes at `0.4`, while
  `0.6+` fails loss. At `r88`, uncovered opacity `0.2/0.4/0.6/0.8` passes and
  moves alpha `>0.1` `0.414/0.416/0.416/0.417`, normal PSNR
  `5.729/5.756/5.771/5.782`, and oracle `25.242/25.032/24.897/24.802`;
  low-alpha passes through `0.6` but fails at `0.8`. The read: scalar born
  opacity is another smooth coverage/oracle tradeoff and does not justify long
  continuation. Next gate should change support shape, such as anisotropic
  birth support along the fitted trajectory.
- The anisotropic birth-support gate is now run and is a clean negative:
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_anisotropic_support_sweep_cap128.md`
  and
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_anisotropic_support_sweep_cap128_dense_support.md`.
  The implementation adds `support_birth_split.support_shape` with
  `trajectory_ellipse`, plus along/across/precision radii, and the sweep
  harness can grid those fields. The 8-row gate uses `32` births, cap `128`,
  along `88px`, across `24/32px`, precision `48px`, opacity `0.4/0.6`, and
  both uncovered-brightness and low-alpha target sources. Every row passes with
  zero overflow (`100/71/128` max/p95/cap), but alpha `>0.1` stays
  `0.408-0.409`, below the prior isotropic uncovered row at `0.411`; forced
  alpha stays `14.554-14.566` and oracle stays `25.404-25.541`. The read:
  one fitted global ellipse is too coarse for a broad target field. Do not
  expand this exact grid; next support work should use multi-center or
  stratified birth/split, then rerun radius/opacity/cap sweeps on that primitive.
- Multi-center birth/split is now the first real coverage move after the
  single-center negatives:
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_support_sweep_cap128.md`
  and
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_support_sweep_cap128_dense_support.md`.
  The implementation adds `support_birth_split.center_strategy=farthest_xy` and
  `center_count`, while the default `global_line/c1` preserves old behavior.
  The 4-row cap-128 gate uses `32` births, radius `64`, target sources
  `uncovered_brightness/low_alpha`, and center counts `4/8`. Best row is
  `uncovered_brightness_n32_r64_farthest_xy_c8_cap128`: pass true, zero
  overflow, max/p95/cap `101/71/128`, `181.1ms` step, `63.5ms` backward,
  alpha `>0.1` `0.4309`, alpha `>0.5` `0.1550`, normal PSNR `5.843`,
  forced-alpha PSNR `14.608`, oracle `23.965`. That beats the prior cap-128
  coverage rows (`0.411` baseline, `0.417` r88, `0.420` low-alpha r96) but
  gives back oracle. Next sweep should stay on multi-center K=8 and vary radius
  and opacity to recover oracle without losing the coverage gain.
- The multi-center K8 radius/opacity sweep is now run:
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_radius_opacity_sweep_cap128.md`
  and
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_radius_opacity_sweep_cap128_dense_support.md`.
  It sweeps uncovered-brightness, `K=8`, `32` births, cap `128`, radii
  `48/56/64/72`, and opacities `0.4/0.6/0.8`. All 12 rows pass with zero
  overflow and max/p95/cap `100-101/71/128`. Best coverage is `r72/o0.8` with
  alpha `>0.1` `0.431797`, alpha `>0.5` `0.163643`, normal PSNR `5.871`,
  forced-alpha `14.605`, but oracle `23.670`. The selected balanced row is
  `r64/o0.4`: alpha `>0.1` `0.429806`, alpha `>0.5` `0.138504`, normal PSNR
  `5.789`, forced-alpha `14.620`, oracle `24.805`, mean step/backward
  `167.9/58.1ms`. It keeps almost all K8 coverage while recovering `+0.84`
  oracle PSNR versus the prior `r64/default` row. Next gate should be a short
  20-step media run on `K=8/r64/o0.4` before longer continuation.
- The selected `K=8/r64/o0.4` 20-step media gate is now run:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_20step_media.jsonc`,
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_media.json`,
  and
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_o04_20step_dense_support.md`.
  It passes with zero overflow (`101/71/128`), weighted loss
  `0.903197 -> 0.897231`, feature loss `0.631571 -> 0.631083`, probe PSNR
  `21.681 -> 21.769`, full RGB PSNR `5.794`, mean step/backward
  `157.5/59.3ms`, and last step/backward `147.3/54.3ms`. Dense support after
  20 steps holds the coverage gain: alpha `>0.1` `0.431158`, alpha `>0.5`
  `0.138506`, forced-alpha `14.631`, oracle `24.851`. This is a positive
  support gate, but not a visual-quality solution; the media still shows a
  sparse/black render, so do not scale to 300 clips yet.
- The matched `K=8/r72/o0.4` 20-step comparison is now run:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r72_o04_from1500_lr001_20step_media.jsonc`,
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r72_o04_20step_media.json`,
  and
  `outputs/benchmarks/2026-05-20_star_uvt_birthsplit_multicenter_k8_r64_vs_r72_o04_20step_dense_support.md`.
  It passes with zero overflow (`101/71/128`), weighted loss
  `0.910099 -> 0.903088`, feature loss `0.633414 -> 0.632829`, probe PSNR
  `21.601 -> 21.703`, full RGB PSNR `5.820`, mean step/backward
  `157.9/61.1ms`, and last step/backward `140.3/53.0ms`. Dense support is
  slightly higher than `r64/o0.4` (alpha `>0.1` `0.432454`, alpha `>0.5`
  `0.146591`, forced-alpha `14.635`), but target-background oracle drops to
  `24.668` and feature/probe losses are worse. This kept `K=8/r64/o0.4` as the
  balanced 20-step default; the later 50-step continuation and cap-safe
  pressure reduction are recorded below.
- The 50-step `K=8/r64/o0.4` continuation is now materialized as a concrete
  config and preflight gate:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_50step_media.jsonc`,
  `research_experiments/star_uvt_feature_tubes/preflight_support_birthsplit_gate.py`,
  and
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_r64_o04_50step_preflight/summary.md`.
  The RGB-probe/colorizer checkpoint and V-JEPA feature cache have now been
  regenerated locally. The probe row is
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.json`;
  it writes
  `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.pt`
  and cached target features
  `outputs/feature_cache/star_uvt_feature_targets/vjepa2_1_vitb_256crop_64f/a524619cf73c9cc18bdbe53d.pt`.
  It passes with grid loss `0.044358 -> 0.004494`, final full PSNR `20.089`,
  and offline W&B run `onsehts5`.
  The STAR checkpoint-ladder has also been regenerated through 1300 steps.
  First segment:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_300step_checkpoint.jsonc`
  produced
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step.pt`.
  It passes with loss `1.458365 -> 1.057802`, feature loss
  `0.999935 -> 0.812539`, RGB-probe PSNR `13.387 -> 16.104`, zero tile
  overflow, mean step/backward/render `3975.9/1983.0/1438.0ms`, and offline
  W&B run `alkbeo34`. The 300->600 segment
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_resume300_from300_checkpoint_media.jsonc`
  produced
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_600step_after_resume.pt`.
  It passes with loss `1.056025 -> 0.752422`, feature loss
  `0.811725 -> 0.654100`, RGB-probe PSNR `16.121 -> 20.074`, and zero tile
  overflow. The 600->800 segment
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.jsonc`
  produced
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_800step_after_resume.pt`.
  It passes with total loss `0.556334 -> 0.403675`, RGB-probe PSNR
  `20.078 -> 22.458`, and zero tile overflow; feature loss intentionally
  carries only weight `0.25` there and increases `0.653852 -> 0.706235`, so do
  not cite it as a feature-loss win. The 800->1000 segment
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_checkpoint_media.jsonc`
  produced
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_1000step_after_resume.pt`
  plus RGB-probe media. It passes with loss `0.762971 -> 0.428924`, feature
  loss `0.706284 -> 0.637935`, RGB-probe PSNR `22.465 -> 22.598`, zero tile
  overflow, mean step/backward/render `3368.5/1836.8/1137.3ms`, and offline
  W&B run `bubca3vm`. The 1000->1100 segment
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.jsonc`
  produced
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_1100step_after_resume.pt`
  plus RGB-probe media. It passes with total loss `0.538652 -> 0.503427`,
  RGB-probe PSNR `22.602 -> 23.537`, zero tile overflow, mean
  step/backward/render `3433.4/1927.0/1125.4ms`, and offline W&B run
  `pvv0mbwo`; feature loss increases `0.637887 -> 0.652565` under weight
  `0.5`, so cite it as a total/RGB-probe win, not a feature-loss win. The
  1100->1200 segment
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.jsonc`
  produced
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_1200step_after_resume.pt`
  plus RGB-probe media. It passes with loss `0.740994 -> 0.600747`, feature
  loss `0.652525 -> 0.624458`, RGB-probe PSNR `23.542 -> 23.552`, zero tile
  overflow, mean step/backward/render `3443.2/1934.2/1128.2ms`, and offline
  W&B run `08458lgu`. At that intermediate point, the 50-step support
  preflight still blocked only on the final sparse-forward 1500-step resume
  checkpoint. The 1200->1250 segment
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_checkpoint_media.jsonc`
  produced
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_1250step_after_resume.pt`
  plus RGB-probe media. It passes with total loss `0.644657 -> 0.636518`,
  RGB-probe PSNR `23.557 -> 23.817`, zero tile overflow, mean
  step/backward/render `4591.2/2448.5/1450.6ms`, and offline W&B run
  `y0ml2jc9`; feature loss increases `0.624403 -> 0.627228` under weight
  `0.75`, so cite it as a total/RGB-probe win, not a feature-loss win. The
  1250->1300 segment
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.jsonc`
  produced
  `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_1300step_after_resume.pt`
  plus RGB-probe media. It passes with loss `0.793051 -> 0.775637`, feature
  loss `0.627185 -> 0.618493`, RGB-probe PSNR `23.823 -> 24.058`, zero tile
  overflow, mean step/backward/render `4806.3/2630.3/1474.5ms`, and offline
  W&B run `fkjzpli1`. The remaining sparse-forward ladder has now been
  regenerated. The 1300->1400 sparse-forward/batched-VJP segment
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_sparseforward_batchedvjp_checkpoint_media.jsonc`
  writes
  `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_1400step.pt`
  and
  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_from1300_100step_media.json`;
  it passes with loss `0.775389 -> 0.757040`, feature loss
  `0.618394 -> 0.609855`, RGB-probe loss `0.003925 -> 0.003680`, RGB-probe
  PSNR `24.342`, zero overflow, mean step/backward/render
  `722.5/328.6/213.8ms`, and offline W&B run `inu9e86f`. The 1400->1450
  lr001 segment writes
  `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1400_lr005sparse_1450step.pt`
  and passes with total loss `0.756800 -> 0.756539`, RGB-probe PSNR `24.366`,
  zero overflow, and mean step/backward/render `859.4/363.5/272.5ms`, but
  feature loss slightly worsens `0.609756 -> 0.610156`. The 1450->1500 segment
  writes
  `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_1500step.pt`
  and passes with loss `0.756490 -> 0.752234`, feature loss
  `0.610136 -> 0.608145`, RGB-probe PSNR `24.434`, zero overflow, mean
  step/backward/render `983.7/381.2/311.8ms`, and offline W&B run `hlo6xs7x`.
  The 50-step `K=8/r64/o0.4` preflight is now clean `ready`, and the support
  run has also been executed:
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_r64_o04_50step_media.json`.
  It is a useful failed gate, not a promotion: `pass=false` because
  `tile_overflow_sum=277`, max tile count `146/128`, and overflow excess refs
  `1233`. The loss/probe direction is positive despite the overflow:
  weighted loss `0.773832 -> 0.760400`, feature loss `0.612675 -> 0.611403`,
  RGB-probe loss `0.004029 -> 0.003725`, and RGB-probe PSNR
  `23.948 -> 24.289`. The selected 32-birth r64/o0.4 row should not be scaled;
  next support work should reduce cap-128 support pressure or use cap-256 only
  as a budget diagnostic. That follow-up has now been run from the same 1500
  checkpoint. `K=8/n16/r48/o0.4`
  (`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k8_r48_o04_from1500_lr001_50step_media.jsonc`,
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n16_r48_o04_50step_media.json`)
  improves loss `0.757862 -> 0.750863`, feature loss
  `0.609050 -> 0.608136`, and RGB-probe PSNR `24.294 -> 24.476`, but still
  fails fixed-bin by two overflow tiles (max `131/128`). `K=8/n16/r40/o0.4`
  (`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k8_r40_o04_from1500_lr001_50step_media.jsonc`,
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n16_r40_o04_50step_media.json`)
  is the same budget failure (`2` overflow tiles, max `131/128`) while slightly
  improving endpoint losses (`0.756313 -> 0.750070`, feature
  `0.608773 -> 0.607858`, RGB-probe PSNR `24.331 -> 24.491`). The valid row is
  `K=8/n8/r40/o0.4`
  (`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_50step_media.jsonc`,
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_50step_media.json`):
  `pass=true`, zero overflow, max tile `123/128`, fixed-bin eligible, loss
  `0.754568 -> 0.749460`, feature loss `0.608402 -> 0.607554`, RGB-probe loss
  `0.003654 -> 0.003548`, RGB-probe PSNR `24.372 -> 24.501`, dense RGB PSNR
  `6.472`, and offline W&B run `xc3rv44y`. The dense diagnostic
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_pressure_50step_dense_support.md`
  says `n8/r40` improves over `start1500` (`6.035 -> 6.472` normal PSNR,
  `10.702 -> 14.018` forced-alpha, `16.787 -> 21.602` oracle), but it is still
  coverage/composition limited; invalid higher-pressure rows have slightly
  stronger normal/forced support, so use `n8/r40` as the current safe seed, not
  a visual closeout. The longer safe-row promotion gate then selects the
  90-step checkpoint:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_90step_checkpointselect_media.jsonc`,
  `outputs/checkpoints/2026-05-25_star_uvt_feature_targetgrid_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_90step_checkpointselect.pt`,
  and
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_90step_checkpointselect_media.json`.
  It passes with zero overflow, max tile `122/128`, loss
  `0.754568 -> 0.747006`, feature loss `0.608402 -> 0.606764`, RGB-probe loss
  `0.003654 -> 0.003506`, RGB-probe PSNR `24.372 -> 24.552`, dense RGB PSNR
  `6.462`, mean step/backward/render `837.9/373.8/250.9ms`, and offline W&B
  run `3821f8dh`. The matching 100-step overrun
  (`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_100step_media.jsonc`,
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_100step_media.json`)
  remains fixed-bin (`0` overflow, max tile `122/128`) but fails
  `require_loss_decrease` after late jumps at global steps `1590` and `1594`,
  ending loss `0.755682`, feature loss `0.610522`, and RGB-probe PSNR
  `24.402` (offline W&B `iy24tfn5`). The dense diagnostic
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_n8_r40_90_100step_dense_support.md`
  says dense support is nearly flat across 50/90/100: normal PSNR
  `6.472/6.462/6.450`, forced-alpha `14.018/14.012/14.054`, oracle
  `21.602/21.579/21.681`, and alpha `>0.1` `0.6542/0.6523/0.6506`. Treat the
  90-step row as the current cap-safe checkpoint, but not as a quality
  closeout. The checkpoint-aware tail schedule has now been measured too:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_tail00025_100step_media.jsonc`
  writes
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_tail00025_100step_media.json`
  and
  `outputs/checkpoints/2026-05-25_star_uvt_feature_targetgrid_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_tail00025_100step.pt`.
  It passes with zero overflow, max tile `122/128`, final loss `0.749454`,
  feature loss `0.608167`, RGB-probe PSNR `24.520`, and offline W&B
  `omnvnem7`. The schedule avoids the constant-LR catastrophe but still loses
  to the selected 90-step objective/probe checkpoint. The matching dense
  diagnostic
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_n8_r40_schedule_dense_support.md`
  reports the scheduled row at normal/forced/oracle PSNR
  `6.462/14.012/21.578`, essentially identical to the 90-step row. Next work
  should change support selection/visibility/model handoff, not run another
  schedule-only or broad radius/opacity sweep. The allocation follow-up confirms
  that this is not just proportional center packing. `support_birth_split` now
  has an opt-in `tube_allocation="uniform"` mode. Uniform `K=8/n16/r40/o0.4`
  (`src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k8_r40_o04_uniform_from1500_lr001_50step_media.jsonc`,
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n16_r40_o04_uniform_50step_media.json`)
  distributes tubes `[2,2,2,2,2,2,2,2]` but still fails by two overflow tiles
  (max `131/128`) while improving loss `0.758282 -> 0.751447`. The
  one-tube-per-center rows `K=16/n16/r40/o0.4` and `K=16/n16/r32/o0.4`
  (`outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_50step_media.json`,
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k16_n16_r32_o04_50step_media.json`)
  also fail by the same two overflow tiles with max `131/128`, so more centers
  and smaller radius do not clear the saturated tiles. The cap-safe
  `K=12/n12/r40/o0.4` rows pass:
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k12_n12_r40_o04_50step_media.json`
  reaches max `127/128`, loss `0.753998 -> 0.749098`, RGB-probe PSNR
  `24.396 -> 24.517`, dense RGB PSNR `6.483`; the 90-step checkpointselect row
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k12_n12_r40_o04_90step_checkpointselect_media.json`
  reaches max `126/128`, loss `0.753998 -> 0.749217`, feature loss
  `0.608633 -> 0.608311`, RGB-probe PSNR `24.396 -> 24.531`, dense RGB PSNR
  `6.474`. The dense comparison
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_k12_n12_dense_support.md`
  keeps the selected checkpoint unchanged: `K=12/n12` is a useful cap-safe
  support-pressure datapoint but does not beat the selected `K=8/n8` 90-step
  objective/feature row or move forced-alpha/oracle support.
  The first cap-aware support bridge is now measured too. It adds
  cap-slack-weighted target scoring plus a post-placement tile-overflow repair
  guard. The raw `K=16/n16/r40/o0.4` cap-slack row selected low-load target
  pixels (`selected_tile_load_mean=17.706`, max `36`, slack mean `0.862`) but
  still failed the same broad-footprint wall (`2` overflow tiles, max
  `131/128`, loss `0.755188 -> 0.749640`). Exact-fit repair dropped two born
  tubes and cleared the initial placement overflow, but training drifted to one
  final overflow tile (`129/128`, loss `0.754551 -> 0.749341`). Guarded repair
  (`tile_overflow_repair_guard_refs=2`, max drops `4`) dropped born tubes
  `[37,194,732,1192]`, cleared the initial `4` overflowing tiles / `7` excess
  refs to post-repair max `126`, and the 50-step row passed final fixed-bin:
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_capslack_repair4g2_50step_media.json`
  reaches zero overflow, max `127/128`, loss `0.753847 -> 0.749102`, feature
  `0.608604 -> 0.607608`, RGB-probe PSNR `24.400 -> 24.513`, and dense RGB
  PSNR `6.486`. The dense comparison
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_capslack_repair4g2_dense_support.md`
  reports `capslack_repair4g2_50` at normal/forced/oracle PSNR
  `6.486/14.021/21.571`, alpha `>0.1` `0.655`, essentially a tiny scalar nudge
  over `K=12/n12` and still below the selected `K=8/n8` 90-step objective row.
  This promotes cap-aware placement plus guarded tile repair as a useful bridge
  primitive, not as the selected checkpoint. Next STAR support work should add
  residual/visibility-aware target scoring or a stronger feature/RGB handoff;
  tile slack and repair alone do not close the visibility/composition gap.
  The first residual/visibility-aware scoring row is now measured as a
  mixed-positive/flat result. Added
  `cap_slack_residual_uncovered_brightness`, which renders sparse sampled
  feature values, colorizes them with the frozen probe, scores black-background
  RGB residual where alpha is low, and multiplies by tile slack. The 50-step
  guarded row
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_residualcapslack_from1500_lr001_50step_media.jsonc`
  writes
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_residualcapslack_repair4g2_50step_media.json`:
  `pass=true`, zero overflow, max `127/128`, loss `0.753586 -> 0.748839`,
  feature `0.608503 -> 0.607558`, RGB-probe PSNR `24.404 -> 24.520`, and
  dense RGB PSNR `6.486`. It selected high residual points
  (`selected_residual_mean=0.803`) with low alpha (`0.00618`) and still low-ish
  tile load (`18.826` mean, `58` max); guarded repair again dropped four born
  tubes and cleared the initial `4` overflowing tiles / `7` excess refs to
  post-repair max `126`. The dense comparison
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_residualcapslack_repair4g2_dense_support.md`
  reports normal/forced/oracle PSNR `6.486/14.019/21.579`, alpha `>0.1`
  `0.655`. This is a small scalar objective win over plain cap-slack repair,
  but not a support-geometry win; coverage/composition remains the blocker.
  Local timing for this row is not promotion evidence because the machine was
  contended by unrelated high-CPU jobs during/after the run.
  The footprint-aware residual variant is now measured too. Added
  `cap_slack_footprint_residual_uncovered_brightness`, which mean-pools
  residual/brightness/uncovered score over the projected support radius before
  applying tile slack. The 50-step guarded row
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_from1500_lr001_50step_media.jsonc`
  writes
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_repair4g2_50step_media.json`:
  `pass=true`, zero overflow, max `127/128`, loss `0.752912 -> 0.748672`,
  feature `0.608350 -> 0.607417`, RGB-probe PSNR `24.420 -> 24.521`, and
  dense RGB PSNR `6.481`. It picked broader-footprint targets
  (`footprint_radius_samples=5`, selected residual `0.755`, alpha `0.0549`,
  tile load mean/max `17.698/25`) and used the same guarded four-drop repair.
  The dense comparison
  `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_footprintresidualcapslack_repair4g2_dense_support.md`
  reports normal/forced/oracle PSNR `6.481/14.021/21.576`, alpha `>0.1`
  `0.655`. This is the best scalar loss of the K16 cap-safe bridge rows, but
  dense support is still flat; target picking has mostly exhausted itself
  without an alpha/composition or stronger born-support handoff change.
  The first born-support feature handoff is now measured as a small positive.
  Added `support_birth_split.feature_init_mode=target_group_mean`, which samples
  the normalized target-grid feature at selected birth points and initializes
  each new support center group to its target-feature mean instead of preserving
  the replaced low-opacity tube feature. The guarded 50-step row
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_from1500_lr001_50step_media.jsonc`
  writes
  `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_repair4g2_50step_media.json`:
  `pass=true`, zero overflow, max `127/128`, loss `0.752454 -> 0.748504`,
  feature `0.608332 -> 0.607351`, RGB-probe PSNR `24.433 -> 24.524`, and
  dense RGB PSNR `6.488`. The init was active (`feature_abs_mean`
  `0.123 -> 0.416`) and used the same guarded repair drops `[37,85,194,732]`.
  The dense comparison
  `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_targetinit_repair4g2_dense_support.md`
  improves normal/forced/oracle PSNR to `6.488/14.054/21.629`, while alpha
  `>0.1` remains `0.655`. So target-grid feature init helps content/oracle a
  little, but it does not solve coverage; the next STAR gate should push direct
  alpha/composition or visibility-prefix behavior rather than another target
  picker.
  The first focused support-target alpha bridge is now measured too. Added
  `support_birth_split.target_alpha_loss_weight/target/target_alpha_max_points`
  and used the selected birth target points as a sparse alpha objective
  (`weight=0.25`, `target=0.75`, `2048` samples). The guarded 50-step row
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetalpha_from1500_lr001_50step_media.jsonc`
  writes
  `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetalpha025a075_repair4g2_50step_media.json`:
  `pass=true`, zero overflow, max `127/128`, total loss
  `0.875695 -> 0.868414`, feature `0.608332 -> 0.607645`, RGB-probe PSNR
  `24.433 -> 24.524`, support-target alpha loss `0.492962 -> 0.478448`, and
  dense RGB PSNR `6.508`. The dense comparison
  `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_targetalpha_repair4g2_dense_support.md`
  reports normal/forced/oracle PSNR `6.508/14.084/21.626` and alpha `>0.1`
  `0.657`. This confirms pointwise target-alpha learns and gives a tiny
  coverage/PSNR nudge, but it does not collapse the forced-alpha/oracle gap.
  The first support-target-area bridge is measured too. It uses 2x2 patches
  around `1024` selected birth targets with black-background patch-mean
  composition (`weight=0.5`) and writes
  `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_repair4g2_50step_media.json`:
  `pass=true`, zero overflow, max `127/128`, total loss
  `1.051414 -> 1.040208`, feature `0.608309 -> 0.608125`, RGB-probe PSNR
  `24.433 -> 24.520`, target-area loss `0.597970 -> 0.581641`, and dense RGB
  PSNR `6.507`. The dense diagnostic
  `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_targetarea2_repair4g2_dense_support.md`
  reports normal/forced/oracle PSNR `6.507/14.085/21.627` and alpha `>0.1`
  `0.657`. This is much cheaper than pointwise target-alpha but lands on the
  same support plateau and worsens feature loss versus target-init. The binner
  repair, prefix-tape diagnostic, and prefix-alpha follow-up supersede this as
  the next step: selected support owns sampled target rays and local
  prefix-alpha pressure learns, but dense support remains flat against binfix.
  STAR work should now broaden ownership/coverage or change support sampling
  rather than repeat another pointwise, small-patch, or prefix-pressure variant.
- A fresh matched dynamic-gsplat 512px smoke closes the immediate comparator
  gap at smoke level:
  `outputs/benchmarks/2026-05-20_dynamic_gsplat_512_matched_probe.md`.
  The fixed 512px precomputed-feature dynamic-gsplat config uses
  `64f/512px/8192` active Gaussians and cached V-JEPA conditioning. Step 5
  timing is `8.019s` total, `5.638s` backward, `1.062s` forward decode,
  `0.639s` sample/load, and `0.362s` rasterize. This makes the dynamic-gsplat
  smoke much slower than both selected STAR UVT helper routes at this local
  scale, and it is backward-dominated rather than data-loader- or
  rasterizer-dominated. Treat it as a smoke comparator, not a final quality
  baseline.
- The stronger fixed-512 dynamic-gsplat 20-step media comparator closes the
  local quality/ranking gap enough for current routing:
  `outputs/benchmarks/2026-05-20_dynamic_gsplat_fixed512_20step_matched_media.md`.
  It uses the same `64f/512px/8192` active-Gaussian scale with offline W&B
  media. Loss improves `0.601325 -> 0.492486`, but final eval is still weak:
  PSNR `5.587`, SSIM `0.165`, L1 `0.469`, and the GT-vs-pred media is a
  smeared blob. Mean timed step/backward over steps `5/10/15/20` is
  `2.940/1.926s`; rasterize is only `0.141s` and sample/load `0.292s`. This
  supersedes the 5-step smoke as the current dynamic-gsplat comparator, but it
  is still slower and lower-quality than the compact STAR UVT visual helper.
- `firstclass_backward_breakdown.py` splits the real first-class graph into
  render forward, `FeatureToColor`/loss forward, image-space backward to
  `grad_feature_image`/`grad_alpha`, and Metal renderer backward. It changes the
  512px bottleneck diagnosis: renderer backward is only `22.1%` of backward at
  4096 tubes and `16.9%` at 8192 tubes; `FeatureToColor`/loss backward is the
  larger cost. At 256px/32768t/cap256, renderer backward is about `36%`.
- The first no-pre-norm colorizer A/B is a large speed result but not yet a
  quality promotion: 512px/8192t/chunk2 with `feature_direct_gradcache` and
  `colorize.pre_norm=false` passes a 2-step trainer gate at `3.715s/step`,
  `1.586s` backward, and zero overflow, versus the default pre-norm row at
  `7.937s/step`, `4.883s` backward.
- The 2026-05-19 20-step media A/B keeps the same conclusion: both rows pass
  finite/gradient/no-overflow and write contact sheets plus MP4s. No-pre-norm
  is faster (`7.366s/step`, `3.370s` backward) than default pre-norm
  (`11.098s/step`, `7.070s` backward), but it ends slightly worse
  (`0.32053` loss / `4.941` PSNR versus `0.31742` / `4.984`). Treat it as the
  fast candidate only.
- The checked launch helper for that fast candidate is
  `src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh
  star-feature-512-fast`, backed by
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_sparseforward_batchedvjp_checkpoint_media.jsonc`.
  The older RGB-target speed row is still available through
  `star-feature-512-rgbfast`. The helper now also exposes
  `star-feature-512-visual` for compact target-area visual overfit
  (`930.6ms`, `6.023` full RGB on the current-build helper gate) and
  `star-feature-512-native-fullcell` for the promoted exact full-support native
  vec4 W^T baseline. Compact native star-only is not a keeper because it
  freezes the colorizer and is slower than compact autograd.
- Gate 4 same-clip quality bracket now exists and fails feature promotion:
  RGB STAR direct-atomic reaches `12.444` PSNR after 20 steps on the same
  64f/512px/8192t test-video bracket, while the best feature STAR row reaches
  only `4.987` PSNR after the hidden-64 diagnostic. Dynamic RGB and projected
  F32 rows in this report are
  speed-only synthetic references, not quality comparators.
- The identity/no-pre-norm decoder diagnostic removes both sigmoid and pre-norm
  and is the fastest 512px feature row so far (`2.536s/step`, `1.173s`
  backward), but it ends worse (`0.32446` loss / `4.888` PSNR). This rules out
  the simple "remove decoder clamps/norms" hypothesis as a quality fix.
- The hidden-64 pre-norm decoder-capacity diagnostic is a negative practical
  result: it barely improves best feature PSNR (`4.984 -> 4.987`) while slowing
  to `19.180s/step` and `13.769s` backward. Naive per-pixel decoder capacity is
  not the Gate 4 quality bridge.
- The pre-norm sigmoid gain-2 colorizer-init diagnostic is also negative:
  `4.987` PSNR at `14.119s/step` and `8.913s` backward. It is a tiny PSNR
  change for worse speed than gain-4 linear pre-norm.
- The first-class `src/train/train.py` bridge exists as
  `arch=star_uvt_feature_overfit`, with a checked-in 8f/64px video smoke config.
- Frame chunking preserves global STAR time by shifting `ma.z` per chunk. It is
  a memory valve for 64f/512px F32, not a tiny-run speed win.

Key docs/artifacts:

- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py`
- `research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py`
- `research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py`
- `research_experiments/star_uvt_feature_tubes/feature_autograd_overfit_benchmark.py`
- `research_experiments/star_uvt_feature_tubes/firstclass_scale_report.py`
- `research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py`
- `research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_bridge_audit.py`
- `research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_vs_gaussian_comparison.py`
- `research_experiments/star_uvt_feature_tubes/target_cache_budget.py`
- `research_experiments/renderer_scaling_report.py`
- `src/train/train_star_uvt_feature_overfit.py`
- `src/train_configs/star_uvt_feature_testvideo_8f_64_directatomic_20step.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_8f_64_rgbpyramid_target_gradcache_reduce_vec4_10step.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_8f_64_vjepa_target_gradcache_reduce_vec4_10step.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_chunkedtarget_lr005_5step.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_cachedchunks_lr005_5step.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_20step_media.jsonc`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_fast_overfit_reduce_vec4_summary.md`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_selected_shader_scale_128_256_512.md`
- `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix.md`
- `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix_repeat_top.md`
- `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_pixel_vjp_report.md`
- `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_pixel_vjp_profile.md`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsepixvjp_64f512_from1300_5step.json`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsepixvjp.jsonc`
- `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_grid_vjp_report.md`
- `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_grid_vjp_profile.md`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsegridvjp_64f512_from1300_5step.json`
- `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparsegrid_render_mode_trainer_matrix.md`
- `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparsegrid_render_mode_trainer_matrix.json`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsegridvjp.jsonc`
- `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md`
- `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_rgbpyramid_target_bridge_smoke.md`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_rgbpyramid_target_8f_64px_512t_f32_gradcache_reduce_vec4_10step.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_bridge_smoke.md`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_8f_64px_512t_f32_gradcache_reduce_vec4_10step.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_512_scale_gate.md`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_chunkedtarget_lr005_5step.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_cachedchunks_lr005_5step.json`
- `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`
- `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.json`
- `outputs/benchmarks/2026-05-19_star_uvt_target_cache_budget.md`
- `outputs/benchmarks/2026-05-19_star_uvt_target_cache_budget.json`
- `outputs/benchmarks/2026-05-19_renderer_scaling_report.md`
- `outputs/benchmarks/2026-05-19_renderer_scaling_report.csv`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_20step_media.json`
- `agent_notes/loose_notes/2026-05-19_03-14-36_star_uvt_reduce_vec4_fast_overfit_gate.md`
- `agent_notes/loose_notes/2026-05-19_03-29-14_star_uvt_selected_shader_scale_gate.md`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_gate0_dense_contract_mps_repeat.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_tiny_parity.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_64f_128_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_directatomic_serial_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun2_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_tiny_parity.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_serial_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun3_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_same_session_before_cachedbins_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_cachedbins_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_fused_first3_sigmoid_mse_tiny_parity.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_fused_first3_sigmoid_mse_serial_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun4_after_fused_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_rerun2_after_fused_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_tiny_parity.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_serial_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_serial_rerun2_after_skip_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_skip_colorizer_grad_tiny_parity.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_skip_colorizer_grad_serial_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_skip_colorizer_grad_serial_rerun2_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun5_after_linear_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_logit_handoff_tiny_parity.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_logit_handoff_serial_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun6_after_logit_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_vec4_tiny_parity.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_vec4_serial_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun7_after_vec4_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_serial_rerun2_after_vec4_sequential_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_rerun3_after_vec4_sequential_64f_256_32768_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_direct_mode_matrix_128_256_64f_32768t_f32/summary.md`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_overfit_4f_32px_64t_f32.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_overfit_4f_32px_64t_f32_chunkparity.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_autograd_video_overfit_8f_64px_512t_f32_20step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_20step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_20step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_8192t_f32_chunk4_3step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_16384t_f32_chunk4_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_16384t_f32_chunk4_cap256_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_cap256_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_64_cap256_20step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_32_cap256_20step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_96_cap256_20step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_80_cap256_20step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_cap256_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_vec4_alpha1_72_cap256_20step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_2048t_f32_chunk2_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_4096t_f32_chunk2_gradcache_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_2step.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_cachedbins_prenorm_2step.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_256px_32768t_alpha1_72_cap256.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_256px_32768t_alpha1_72_cap256.md`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_repeat2.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_repeat2.md`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_no_prenorm_repeat2.json`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_backward_breakdown_512px_gradcache_4096_8192t_no_prenorm_repeat2.md`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_2step.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_20step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_20step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_identity_no_prenorm_20step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_hidden64_prenorm_20step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_gain2_20step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_directatomic_prenorm_20step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_rgb_testvideo_64f_512px_8192t_directatomic_20step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.md`
- `outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.json`
- `outputs/benchmarks/2026-05-19_renderer_scaling_report.md`
- `outputs/benchmarks/2026-05-19_renderer_scaling_report.csv`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_report.md`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_20step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_20step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_identity_no_prenorm_20step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_hidden64_prenorm_20step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_gain2_20step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_directatomic_prenorm_20step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_rgb_testvideo_64f_512px_8192t_directatomic_20step_contact.jpg`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_20step_sbs.mp4`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_no_prenorm_20step_sbs.mp4`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_identity_no_prenorm_20step_sbs.mp4`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_hidden64_prenorm_20step_sbs.mp4`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_prenorm_gain2_20step_sbs.mp4`
- `outputs/media/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_directatomic_prenorm_20step_sbs.mp4`
- `outputs/media/2026-05-19_star_uvt_rgb_testvideo_64f_512px_8192t_directatomic_20step_sbs.mp4`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_no_prenorm_2step.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_prenorm_20step_media.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_no_prenorm_20step_media.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_identity_no_prenorm_20step_media.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_hidden64_prenorm_20step_media.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_chunk2_8192t_prenorm_gain2_20step_media.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_cachedbins_chunk2_8192t_prenorm_2step.jsonc`
- `src/train_configs/star_uvt_feature_testvideo_64f_512_directatomic_chunk2_8192t_prenorm_20step_media.jsonc`
- `src/train_configs/star_uvt_rgb_testvideo_64f_512_directatomic_8192t_20step_media.jsonc`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_report.md`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_scale_summary.json`
- `outputs/benchmarks/2026-05-18_renderer_scaling_report.md`
- `outputs/benchmarks/2026-05-18_renderer_scaling_report.csv`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_contact.png`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_8f_64px_512t_f32_chunk2_side_by_side.mp4`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step_contact.png`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_alpha1_72_cap256_20step_side_by_side.mp4`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step_contact.png`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_featurefixedbin_alpha1_72_cap256_20step_side_by_side.mp4`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step_contact.png`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_alpha1_72_cap256_20step_side_by_side.mp4`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step_contact.png`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_testvideo_64f_256px_32768t_f32_chunk4_gradcache_reduce_alpha1_72_cap256_20step_side_by_side.mp4`
- `agent_notes/loose_notes/2026-05-18_20-01-13_star_uvt_feature_gate0_contract.md`

Gate 0 result:

- CPU and MPS contracts pass for `[N,F] -> [T,F,H,W] + alpha -> FeatureToColor
  -> RGB loss`.
- CPU chunked parity: loss diff `7.45e-09`, max grad diff `3.73e-09`.
- MPS chunked parity: loss diff `3.73e-09`, max grad diff `1.21e-08`.
- Tiny overfit decreases on both devices.
- Direct Metal F=4/F=32 tiny parity passes. F32 max errors are feature
  `1.43e-06`, ma `7.15e-07`, opacity `9.54e-06`, q `1.91e-06`.
- Direct Metal `64f/256px/32768/F32` timing is finite with zero overflow:
  `757.9ms` total, `190.1ms` forward, `567.8ms` backward.
- Direct Metal `64f/128px/32768/F32` is finite but overflows `8093` tiles, so
  it is a stress/invalid-support row rather than a quality row.
- Direct feature autograd wrapper matches manual direct backward on the small
  parity scene. The real-video mini overfit on
  `test_data/test_video_small_128_4fps.mp4` at 8f/64px/512t/F32 improves loss
  `0.18671 -> 0.04197` and PSNR `7.29 -> 13.77` in 20 steps, with zero overflow.
- Frame-chunked autograd parity passes for chunk size 2: max errors are feature
  `8.35e-07`, ma `1.64e-07`, opacity `2.24e-07`, q `2.98e-07`.
- The first-class trainer/config path passes the same 8f/64px/512t/F32 real
  video smoke. Full-frame loss improves `0.18602 -> 0.04167`; chunked
  `frame_chunk_size=2` improves `0.18602 -> 0.04167`, PSNR `7.30 -> 13.80`,
  mean step `76.79ms`, last step `59.51ms`, overflow `0`, tile max `76`,
  p95 `74`, and writes a contact sheet plus side-by-side MP4.
- First-class 64-frame scale probes pass with zero overflow:
  `64f/256px/8192t/F32/chunk4` improves loss `0.32612 -> 0.31141` in 3 steps
  with mean step `964.66ms`, forward `120.89ms`, colorize/loss `71.80ms`,
  backward `736.03ms`, tile max `80`, p95 `63`; `64f/512px/2048t/F32/chunk2`
  improves `0.34517 -> 0.34406` in 2 steps with mean step `4020.73ms`,
  forward `586.93ms`, colorize/loss `281.59ms`, backward `3070.12ms`, tile
  max `11`, p95 `5`.
- 64f/256px higher-capacity diagnostics are not valid quality rows under the
  current cap: `16384t/chunk4` overflows `736` tiles and `32768t/chunk4`
  overflows `8160` tiles, although both still decrease loss over 2 steps.
  The tile diagnostics are the important signal: `16384t` has max tile load
  `151` and p95 `123` against the current 128-entry cap, while default
  `32768t` has max `274` and p95 `238`.
- `STAR_UVT_TILE_CAPACITY=256` changes the validity boundary: `16384t` becomes
  zero-overflow; unpruned `32768t` still overflows `216` tiles; `32768t` with
  support pruning becomes zero-overflow. The current best passing 20-step
  feature-tube candidate is `32768t/alpha>=1/72/cap256`: loss
  `0.31889 -> 0.29290`, PSNR `4.96 -> 5.33`, mean step `1320.92ms`, backward
  `1021.20ms`, max tile `252`, p95 `209`. `alpha>=1/80` and `alpha>=1/96`
  have slightly better 20-step loss (`0.29237` and `0.29150`) but overflow
  late, so they are diagnostics rather than fixed-bin candidates.
- The `feature_direct_fixedbin` mode-contract row is now explicitly an
  eligibility/fallback row, not a kernel row: unpruned `32768t/cap256` records
  `mode_fallback_required=true` after `216` overflow tiles, while
  `32768t/alpha>=1/72/cap256` has zero overflow but still uses the direct
  feature kernel. New trainer outputs record `kernel_backward_mode` and
  `requested_fixedbin_is_direct_atomic_alias` so future reports do not mistake
  fixedbin eligibility for an optimized fixedbin shader.
- The gradcache A/B is a small real win, not the final fast path. Serial
  synthetic `64f/256px/32768t/F32` improves backward `485.63ms -> 471.29ms`.
  First-class `32768t/alpha>=1/72/cap256` records
  `effective_render_mode=feature_direct_gradcache`, zero overflow, mean step
  `1226.04ms`, and backward `973.24ms`.
- The feature-gradient atomic diagnostic is the bigger signal: with the same
  gradcache path but `grad_feature` atomics skipped, synthetic backward drops to
  `327.71ms`; a nearby full-gradcache rerun measured `592.54ms`. The exact
  ratio is timing-noisy, but the direction is strong enough to make
  feature-gradient reduction / RGB-grad handoff the next shader target.
- The first trainable reduction attempt is a negative speed result. The
  `gradcache_reduce_feature_grad` synthetic row passes full F4/F32 parity but
  measures `523.77ms` backward versus a fresh same-session gradcache rerun at
  `491.07ms`. The first-class row also trains (`0.31889 -> 0.29290`, zero
  overflow) but is slower than gradcache (`1260.62ms/step`, `1000.33ms`
  backward versus `1226.04ms/step`, `973.24ms` backward). Do not make this the
  default; the next attempt needs a different reduction shape.
- The vectorized trainable reduction follow-up is also not a first-class win.
  `gradcache_reduce_feature_grad_vec4` passes full F4/F32 parity and improves a
  synthetic direct-kernel control (`484.71ms` backward versus same-session
  gradcache `528.22ms` and scalar reduce `516.44ms`), but the real cap256
  first-class row is slower than both controls (`2094.83ms/step`, `1412.54ms`
  backward versus gradcache `1806.98ms/step`, `1333.10ms` backward and scalar
  reduce `1889.73ms/step`, `1394.79ms` backward). Treat it as a diagnostic
  proving scalar channel reduction was not the real trainer bottleneck.
- The 512px tube-count bracket now reaches 4096 and 8192 tubes with zero
  overflow under `feature_direct_gradcache`, but the timing is not usable for a
  large quality run: 4096t is `6456.35ms/step` with `4208.94ms` backward, and
  8192t is `7937.28ms/step` with `4882.82ms` backward plus `1223.41ms`
  color/loss. Do not launch 512px/32768t until the backward or colorize/loss
  path changes.
- The narrow RGB handoff prototype is the positive direction signal:
  `fused_first3_sigmoid_mse` passes F4/F32 parity and measures `309.31ms`
  backward on the same synthetic target, versus `547.58ms` for a same-session
  full gradcache rerun and `351.58ms` for a same-session skip-feature-gradient
  diagnostic. It only covers fixed first-three-channel sigmoid MSE; the
  generalized linear follow-up now passes parity but is slower than gradcache.
- That generalized in-tile handoff has now been tested and should not be
  promoted as-is: `direct_linear_sigmoid_mse_backward` passes parity including
  colorizer weight/bias grads, but measures `618.55ms` and `615.55ms` backward
  on two target runs, slower than the same-session gradcache rerun at
  `477.50ms`. Skipping colorizer grads did not yield a stable mean win
  (`714.13ms`, then `598.45ms` backward), so the next handoff should keep some
  colorizer/loss reduction in image space or use a different renderer-side
  accumulation shape.
- The cheaper image-space-prep logit handoff was also a negative speed result:
  `direct_logit_handoff_backward` passes parity, but measures `595.15ms`
  renderer backward plus `60.17ms` handoff prep (`835.63ms` total), while a
  same-session gradcache rerun is `528.96ms` backward and `693.23ms` total.
  Replacing the dense F-channel gradient image with RGB/logit gradients does
  not remove the per-pixel `W^T` and per-channel feature-gradient atomic costs.

Next useful experiment:

- Treat cap 256 plus `alpha>=1/72` as the quality-max passing 32768-tube
  validity candidate, and keep `alpha>=1/64` as the conservative fallback.
- Use `feature_direct_fixedbin` in configs when checking promotion eligibility,
  use `feature_direct_gradcache` as the current fastest valid feature mode, and
  keep the next implementation target on reducing feature-gradient atomics via
  a non-barrier-heavy accum/reduce path or an optimized fixedbin backward. Do
  not promote the current linear/logit handoff prototypes or the vec4 reducers.
  The new logit-handoff vec4 reducer is the best diagnostic shape so far, but
  it is still synthetic-only and needs a first-class trainer gate.
- Keep 512px feature probes on small tube counts until speed improves; 4096 and
  8192 tubes have support headroom but are already too slow under pre-norm.
- Use the first-class backward breakdown before proposing another shader fork:
  at 512px, optimizing only Metal renderer backward cannot recover most of the
  step time unless the `FeatureToColor`/loss VJP also changes.
- The no-pre-norm 512px/8192t 20-step media A/B converges and is faster, but it
  does not beat pre-norm quality. Keep it as a speed setting while renderer
  fixedbin/tile-slot accumulation and image-space VJP work continues.
- The identity/no-pre-norm decoder is faster again, but quality is worse than
  both sigmoid variants. Treat simple activation/norm removal as closed; next
  quality work should test feature initialization, objective shape, or decoder
  capacity.
- The hidden-64 decoder-capacity test barely moves PSNR and is much slower, so
  avoid larger dense per-pixel decoders as the next default quality route.
- The gain-2 pre-norm init row also barely moves PSNR and slows the row, so
  simple colorizer init gain is not the next default quality route either.
- Gate 4 says feature STAR is not yet a source-overfit quality replacement for
  RGB STAR. Next quality work needs the feature decoder/objective, not another
  renderer-only speed fork.
- Use the regenerated renderer scaling report's `64f/256px/32768` section as
  the comparison surface for RGB STAR, STAR F32 feature, dynamic RGB, and
  projected F32 feature rows.
- Keep 512px on frame chunks and step tube count gradually; the current 2048
  tube row is already backward-dominated at `4.02s/step`.
- Port gradcache/accum/fixedbin modes with explicit overflow fallback.

## Gaussian 300-Clip V-JEPA/Static-Dynamic Scale Lane

Status: **Blocked at 512px promotion**

Purpose: scale the V-JEPA/static-dynamic Gaussian trainer across prepared
single-video windows with cached V-JEPA conditioning.

Current decisions:

- Cached conditioning and prefetch are working at 256px.
- The old huge slowdown was prediction-side V-JEPA feature-loss backward, not
  cache misses. Keep V-JEPA feature loss off unless specifically testing it.
- Do not trust the existing 256->512 multires config as a completed baseline
  until the NaNs after promotion are fixed or guarded.

Key config/logs:

- `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_300clips_3kstep_multires_256to512.jsonc`
- `outputs/run_logs/scale300_prefetch2_8192_multires_profile12_20260517_232500.log`
- `outputs/run_logs/dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143.log`
- `data/feature_cache/single_video_pretrain_300_youtube_vjepa2_1_vitb_256crop_64f_512center_nativefps`
- `agent_notes/loose_notes/2026-05-17_23-19-05_shader_audit_and_fast_overfit_plan.md`

Known facts:

- 300/300 cache files existed in the recorded audit.
- Warm 256px profile with prefetch was about `2.06s/step`.
- The full multires run hit NaNs around the 512px switch.
- A fixed-512 one-record smoke on 2026-05-20 completes but is slow at the
  selected STAR-comparison scale: step 5 is `8.019s` total with `5.638s`
  backward on `64f/512px/8192` active Gaussians.

Next useful experiment:

- Add 512px promotion guardrails: clamp/diagnose fov/camera terms, checkpoint
  before promotion, and resume from a clean pre-promotion checkpoint.

## Mixed Same-View + Heldout Novel-View Trainer

Status: **Active next implementation**

Purpose: train the same model path on broad same-view data and calibrated
multicam heldout supervision without mixing their metrics.

Current decisions:

- `same_view_recon` and `heldout_view_recon` must stay separate in logs.
- The data loaders are ready enough.
- `src/train/mixed_data_scheduler.py` now provides the first typed scheduler
  boundary: same-view and novel-view batch records, explicit loss names,
  shared view sampling, and `both`/`alternate` schedule selection.
- `src/train/train_mixed_same_heldout_implicit_dynamic.py` now consumes that
  boundary for real optimizer steps under
  `arch=mixed_same_heldout_precomputed_feature_implicit_camera`.
- The trainer now calls `mixed_data_scheduler.sample_mixed_step_batch(...)`
  directly instead of duplicating `both`/`alternate` branch logic. The scheduler
  accepts a lazy same-view sequence provider, so novel-only steps keep the
  same-view manifest loader cold. After this slice, the broader focused trainer
  helper suite passed (`70 passed`) and the checked-in mixed smoke passed at
  `wandb/offline-run-20260521_173114-em1oaiqp`.
- A local 2-step offline smoke passed on MPS using the DeepView RGB-pyramid
  multicam smoke plus the local same-view manifest. It exercised same-view at
  step 1 and heldout-view at step 2. Offline W&B:
  `wandb/offline-run-20260521_170331-x5rhxvzo`.
- A checked-in 10-step smoke config now lives at
  `src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc`.
  It passed offline on MPS with W&B dir
  `wandb/offline-run-20260521_170805-5g581zns`. The visible trace alternated
  same-view and heldout-view losses: same-view steps moved roughly
  `0.5239 -> 0.4942`, while heldout-view steps moved roughly
  `0.6087 -> 0.5996`. Treat this as trainer plumbing plus a tiny convergence
  trace, not as evidence that the training math is solved.
- The checked-in 10-step smoke now logs final-step media via
  `logging.always_log_last_step=true`. Current-state rerun:
  `wandb/offline-run-20260521_222750-9yvznqiq`. Evidence in the offline W&B
  record includes `Loss/same_view_recon`, `Loss/heldout_view_recon`, their
  weighted variants, one final `Render_GT_vs_Pred` preview image, and six
  validation videos: TrainView0/TrainView1/Heldout0 rendered plus GT. This
  closes the "media-capable smoke" gap, but it remains a 10-step interface
  trace rather than a benchmark row.
- 2026-05-22 current-state rerun after the artifact/report cleanup passed:
  `wandb/offline-run-20260522_004727-ka4lm8g5`. Launch:
  `PYTHONPATH=src/train WANDB_MODE=offline WANDB_SILENT=true .venv/bin/python src/train/train.py src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc`.
  The run went through `train.py -> trainer_registry ->
  MixedSameHeldoutPrecomputedFeatureTrainer`, used MPS, hit the RGB-pyramid
  feature cache, alternated same-view and heldout-view steps, and finished with
  heldout eval metrics. Offline W&B strings include `Loss/same_view_recon`,
  `Loss/heldout_view_recon`, `TrainView0/Eval/PSNR`,
  `TrainView1/Eval/PSNR`, `Heldout0_camera_0040/Eval/PSNR`, and
  `Heldout/Eval/PSNRMean`. Media artifacts exist for final
  `Render_GT_vs_Pred`, TrainView0/TrainView1 rendered+GT videos, and Heldout0
  rendered+GT videos. This is current interface smoke evidence only; do not add
  a `BASELINES.md` row from it.
- `sequence_data.ManifestSequenceSampler` now owns same-view manifest
  eager/lazy sampling and optional prefetch for both the base token-GS trainer
  and this mixed trainer. The checked-in smoke passed again after that
  extraction at `wandb/offline-run-20260521_171453-7wqptf1i`.
- `RGBReconObjective.require_alpha_for_feature_background(...)` now owns the
  F32 alpha/background safety guard for base, multicam train/heldout, and
  camera-swap paths. A 1-step single-cam F32 smoke passed at
  `wandb/offline-run-20260521_171908-pgv52pgm`; the checked-in mixed smoke
  passed again at `wandb/offline-run-20260521_171924-mkj9af97`.
- `MulticamPrecomputedFeatureImplicitTrainer._recon_loss_for_views(...)` now
  owns the shared multicam train-view and heldout-view reconstruction loop.
  After this extraction, the focused trainer/helper test slice passed
  (`69 passed`) and the checked-in mixed smoke passed again at
  `wandb/offline-run-20260521_172310-9iwq2eer`; a fresh current-state rerun
  passed at `wandb/offline-run-20260521_172710-2xs5airh`. This is still
  plumbing evidence and a tiny loss trace, not a quality or baseline claim.
- `MulticamPrecomputedFeatureImplicitTrainer._rendered_view_recon_loss(...)`
  now shares alpha/background guarding, reconstruction-loss profiling, and
  preview capture across multicam train-view, heldout-view, and camera-swap
  renders. A 1-step oracle-relative camera-swap smoke over the 32px RGB-pyramid
  multicam config passed offline at
  `wandb/offline-run-20260521_173425-bf4yc6h0`. The broader focused trainer
  suite passed (`88 passed`), and the checked-in mixed smoke passed again at
  `wandb/offline-run-20260521_173547-6qpl53pz`.
- `MulticamPrecomputedFeatureImplicitTrainer._step_result(...)` now shares
  result assembly across multicam initial eval, normal train, and camera-swap
  train/eval branches. Validation after this slice: broader focused trainer
  suite `88 passed`; normal 32px multicam smoke
  `wandb/offline-run-20260521_173838-6ittgiyo`; oracle-relative camera-swap
  smoke `wandb/offline-run-20260521_173852-y5kp0ido`; checked-in mixed smoke
  `wandb/offline-run-20260521_173903-yockh84k`.
- `MixedBackwardResult` and `MixedStepAccumulator` now own mixed-step
  aggregation for the same-view/heldout trainer. This removes the parallel
  same-view versus heldout accumulation blocks without collapsing the loss
  names. Validation after this slice: broader focused trainer suite
  `88 passed`; checked-in mixed smoke
  `wandb/offline-run-20260521_174347-hp1dtm6k`.
- `runtime_types.build_step_result(...)` now owns shared `StepResult`
  construction for base token-GS, known-camera, multicam, and mixed
  same-view/heldout paths. Validation after this payload-only slice:
  `py_compile` passed for `runtime_types.py`, base trainer, multicam trainer,
  and mixed trainer; base token-GS smoke passed at
  `wandb/offline-run-20260521_174952-bbbwz3dt`; normal multicam smoke passed at
  `wandb/offline-run-20260521_175020-odhh2imp`; checked-in mixed smoke passed
  at `wandb/offline-run-20260521_175035-4qnp6etn`. This remains plumbing
  evidence, not a quality or convergence claim.
- `pipeline.validation_media.training_preview_payload(...)` now owns the shared
  per-step preview image and optional feature-PCA image payload used by base
  token-GS and relative-pose `val_log` methods. Validation after this media
  payload slice: `py_compile` passed for validation media, base trainer, and
  relative-pose trainer; base token-GS smoke passed at
  `wandb/offline-run-20260521_175514-3nqmd2zg`; normal multicam smoke passed at
  `wandb/offline-run-20260521_175542-6wm1w7v3`; focused relative-pose import
  and config suite passed (`13 passed in 0.96s`). This is still log-path
  plumbing evidence, not convergence evidence.
- `Trainer.run_training_loop(...)` and `print_training_header(...)` now share
  the base/known-camera loop while `KnownCameraTrainer` keeps only
  branch-specific header/export hooks. Validation after this loop slice:
  `py_compile` passed for the base trainer; the temporary 1-step known-camera
  run-loop smoke passed at `wandb/offline-run-20260521_175918-9acf7f5a`;
  `tests/test_temporal_sampling.py tests/test_train_logging.py` passed
  (`15 passed in 1.28s`); the checked-in base token-GS smoke passed at
  `wandb/offline-run-20260521_175933-cv0vixo9`. This confirms the shared
  loop executes, not that training quality improved.
- `Trainer.initial_recon_step_result(...)` now shares the initial eval
  render/reconstruction/V-JEPA/payload path for implicit-camera and known-camera
  trainers. Validation after this initial-eval slice: `py_compile` passed for
  the base trainer; the temporary 1-step known-camera smoke passed at
  `wandb/offline-run-20260521_180254-vgovjn1b`; the checked-in base token-GS
  smoke passed at `wandb/offline-run-20260521_180306-qosbrfqm`. This verifies
  the shared initial payload path runs for both branches.
- `KnownCameraTrainer` now inherits the base `render_full_sequence(...)`
  implementation because its `_eval_decode_clip(...)` override supplies known
  cameras to the shared renderer. Validation after removing the duplicate
  override: `py_compile` passed for the base trainer; a temporary known-camera
  1-step smoke with video logging enabled passed at
  `wandb/offline-run-20260521_180535-tcwvntcd`, exercising the inherited
  full-sequence validation path.
- `MulticamRelativePoseImplicitTrainer` now reuses the inherited
  `_rendered_view_recon_loss(...)` helper for full relative-pose camera-swap
  renders. Validation after this branch cleanup: `py_compile` passed for the
  relative-pose and multicam trainers; a temporary learned-residual
  camera-swap smoke passed at `wandb/offline-run-20260521_180825-m2xarb7g`;
  `tests/test_multicam_relative_pose_trainer.py` passed (`13 passed in
  1.45s`). This is launch/plumbing coverage for the shared loss path, not a
  convergence or visual-quality result.
- The base token-GS trainer now uses `train_logging.init_wandb_run(...)`
  instead of carrying its own `wandb.init(...)` kwargs block. Validation after
  this logging-boundary cleanup: `py_compile` passed for the base trainer and
  `train_logging.py`; config normalization proved old configs default missing
  `logging.wandb_enabled` to `true`; a 1-step base token-GS smoke with the
  legacy default W&B path passed at `wandb/offline-run-20260521_181535-x31edkh2`;
  a second 1-step smoke with `logging.wandb_enabled=false` passed without W&B
  output and ended with the disabled-W&B completion message. This is logging
  contract evidence, not training-quality evidence.
- `MulticamPrecomputedFeatureImplicitTrainer.multicam_validation_payload_from_renders(...)`
  now owns shared validation-video payload assembly for base multicam and
  relative-pose trainers. Validation after this media-boundary cleanup:
  `py_compile` passed for both trainers; a 1-step base multicam smoke with
  video logging passed at `wandb/offline-run-20260521_182035-qluznetc`; a
  1-step relative-pose learned-residual smoke with video logging passed at
  `wandb/offline-run-20260521_182058-pwtxd0j6`. Both smokes encoded the
  multicam validation videos through the shared helper; this is media-path
  plumbing evidence, not convergence evidence.
- `Trainer.temporary_render_size(...)` now owns the generic render-size
  context used by the relative-pose multires path. Validation after this
  render-dispatch cleanup: `py_compile` passed for the base, multicam, and
  relative-pose trainers; a 1-step relative-pose learned-residual smoke with
  validation video logging passed at `wandb/offline-run-20260521_182501-qqrr5zvd`.
  This verifies inherited render-size restore/grid-cache plumbing, not visual
  quality.
- `Trainer.training_preamble_messages(...)` and
  `after_training_complete(...)` now cover lifecycle-only trainer wrappers.
  `PrecomputedFeatureImplicitTrainer` reports feature-cache metadata through the
  preamble hook, and `MulticamRelativePoseImplicitTrainer` saves its optional
  checkpoint through the post-success hook. Validation after this cleanup:
  `py_compile` passed for base/precomputed/multicam/relative-pose trainers,
  `git diff --check` passed on the touched trainer files, and a lightweight
  lifecycle hook smoke passed. This is run-loop plumbing evidence, not
  convergence or math evidence.
- `Trainer.model_eval_mode(...)` now owns eval/train restoration for initial
  diagnostics across base token-GS, known-camera, and multicam trainers.
  Validation after this cleanup: `py_compile` passed for
  base/precomputed/multicam/relative-pose/mixed trainers, `git diff --check`
  passed on the touched trainer files, and a lightweight smoke verified
  train-mode restore, eval-mode preservation, exception restore, and inherited
  access through known-camera and multicam subclasses. This is state-plumbing
  evidence only.
- `Trainer.train_step_context(...)` and `optimizer_step(...)` now share the
  zero-grad/profile/optimizer/timing envelope across base token-GS,
  known-camera, multicam, and mixed same-view/heldout steps. Validation after
  this cleanup: `py_compile` passed for base/precomputed/multicam/relative-pose
  /mixed trainers, `git diff --check` passed on the touched trainer files, and
  a lightweight smoke verified zero-grad, optimizer update, timing payloads, and
  inherited access across the four active trainer classes. This is step-plumbing
  evidence only.
- `Trainer.initial_clip_indices(...)` and `initial_clip_for_sequence(...)` now
  share first-window diagnostic clip setup across base token-GS, known-camera,
  and multicam initial paths. Validation after this cleanup: `py_compile`
  passed for base/precomputed/multicam/relative-pose/mixed trainers,
  `git diff --check` passed on the touched trainer files, and a lightweight
  smoke verified inherited index construction, frame slicing, and the existing
  `(1, F)` `prepare_clip(...)` time-batch contract. This is initial-diagnostic
  plumbing evidence only.
- `KnownCameraTrainer.known_cameras_for_indices(...)` now shares indexed camera
  tuple extraction between known-camera initial eval and full-sequence eval, and
  the train step reuses `sample_clip(...)`. Validation after this cleanup:
  `py_compile` passed for base/precomputed/multicam/relative-pose/mixed
  trainers, `git diff --check` passed on the touched trainer file, and a
  lightweight smoke verified selected camera ordering plus the missing-camera
  error. This is known-camera wiring evidence only.
- `KnownCameraTrainer.sample_clip(...)` no longer overrides the base
  `Trainer.sample_clip(...)` with a four-value camera-aware tuple. The branch
  now uses `sample_known_clip(...)` for training. Validation after this cleanup:
  `py_compile` passed for base/precomputed/multicam/relative-pose/mixed
  trainers, `git diff --check` passed on the touched trainer file, and a
  lightweight smoke verified the four-value known-camera batch plus that
  `KnownCameraTrainer.sample_clip` now resolves to `Trainer.sample_clip`. This
  is trainer-interface evidence only.

Key docs/data:

- `research_notes/data_contract.md`
- `src/train/sequence_data.py`
- `src/train/multicam_video_data.py`
- `data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/train_manifest.jsonl`
- `data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl`
- `data/multicam_val/clip_sets/multicam_val_v1_chunked_128_4fps_16f/manifest.jsonl`
- `src/dataset_configs/multicam_train2_holdout1_5sample_128_4fps_16f_manifest.jsonl`

Next useful experiment:

- Run a longer W&B-enabled mixed trace with media enabled and record separate
  same-view and heldout-view trend rows before promoting this to `BASELINES.md`.

## V-JEPA/F32 Multicam Heldout

Status: **Active benchmark-contract lane**

Purpose: test whether V-JEPA/static-dynamic tokens plus feature splatting can
hold up under heldout-camera evaluation.

Current decisions:

- The promoted F32 goodset alpha threshold is `1/128` for the current setup.
- Current PSNR/SSIM evidence is useful, but pose recovery is not physically
  solved; future rows need pose-error diagnostics and fisheye-preserving A/Bs.

Key configs/W&B:

- `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alphaab_alpha1_128.jsonc`
- W&B `hru1yv0t`: heldout PSNR `13.6248`, SSIM `0.1922`.
- W&B `0pdfypqe`: full relpose RGB goodset.
- W&B `vrr1a8pg`: relpose-only follow-up, negative.
- `BASELINES.md` Tier 2a rows.
- `TODO/vjepa_f32_multicam_heldout_followups.md`

Next useful experiment:

- Promote the lane into a benchmark contract: source/camera-disjoint manifests,
  leakage probes, pose-error diagnostics, fisheye-preserving camera path, and
  explicit `BASELINES.md` rows.

## Browser WebGPU Dynamic Splat Trainer

Status: **Active prototype / source-view browser training smoke green**

Purpose: move the fast local shader work toward a browser-checkable training
surface before attempting full Metal-to-WGSL parity.

Current decision:

- Keep this as a separate SPA prototype at `web/dynaworld_browser_trainer/`.
  It preloads the local Neural3D `coffee_martini` preview from
  `data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/previews/` when
  served from repo root and falls back to a deterministic D-NeRF-style fixture.
- The current WGSL trainer learns a compact dynamic splat/tube residual over a
  mean static background from source-view RGB samples and renders the current
  result live. It is not the STAR UVT Metal tile compositor, not the
  shared-backward/tape accumulation path, not PowerFoam ray walking, and not a
  heldout-camera benchmark.
- The browser prototype now has two selectable approximation modes:
  World Tubes-style shared motion and dynamic-splats-style velocity. This is a
  branch inside the simplified WGSL trainer, not a full port of both native
  Metal shader families.
- Initialization is deterministic target-grid/color seeding from the preloaded
  frames. It does not run COLMAP, pycolmap, VGGT, or any point-cloud geometry
  initializer in the browser.
- The UI defaults to 768 splats and exposes a 96-768 splat-count slider. 512
  was best under the earlier sampler, but the 95% motion-sample retest made 768
  the current quality default. Higher counts should wait for a less quadratic
  browser backward path because the current WGSL objective evaluates all splats
  per sampled pixel.
- `converge16` added deterministic sparse validation metrics. The important
  finding is that global/grid loss can be tiny while moving regions are still
  poor; the motion-weighted validation loss is now the main convergence signal.
- `converge17` added a motion-biased training sampler. The loader packs
  high-energy frame/pixel samples versus the mean background, and the WGSL
  train shader initially used a 75% motion / 25% uniform sample mix.
- `converge18` raises the default learning rate to `0.90` after a short
  browser sweep showed materially better motion-loss improvement than `0.45`,
  and fixes a reset/ResizeObserver race by keeping the global trainer null until
  WebGPU init has completed.
- `converge19` adds motion-aware initialization: most primitives still use the
  aspect-aware target grid, but the last 38% of splats are seeded from the
  high-motion frame/pixel sample buffer with local time centers, tighter radii,
  and slightly higher opacity so they composite over the coarse grid.
- `converge20` changes the visible `Motion Loss` readback from grid-weighted
  validation loss to direct loss over the packed high-motion frame/pixel set.
  This makes the number incomparable with earlier `converge17`-`19`
  motion-loss values, but it is a better truth signal for sparse moving regions.
- `converge21` adds an actual train-step throughput stat (`Steps/s`) and
  aligns the JS trainer fallback learning-rate default with the UI default
  `0.90`. FPS is now only the render-loop rate; `Steps/s` is the optimizer
  progress rate to watch when comparing splat counts.
- `converge22` changes the default splat count from 384 to 512 based on the
  live capacity sweep. 512 nearly matched the 768-splat motion loss while
  staying much faster, and clearly beat the 384-splat default at comparable
  steps.
- `converge23` changes the default temporal support from `0.26` to `0.30`.
  The 512-splat temporal sweep showed narrower `0.18`/`0.22` support starves
  motion gradients, while `0.30` beat `0.26` at matched and longer steps.
- `converge24` exposes that sampler split as a `Motion Mix` slider, and
  `converge25` promotes 95% motion / 5% uniform after the longer 95% probe beat
  the old 75% default at similar step counts.
- `converge26` retests splat capacity under the new sampler and promotes 768
  splats as the current default. The old 512 default was right under the old
  sampler, but it is now capacity-limited.
- `converge27` adds `Motion Cov` and `Active` model-health diagnostics.
  The first diagnostic run showed the old init was lowering motion loss partly
  by reducing dynamic coverage on motion samples.
- `converge28` improves motion-aware initialization: 48% motion-seeded splats,
  slightly broader motion radii, and higher initial motion opacity. This is a
  motion-quality improvement with a small grid-loss tradeoff.
- `converge29`/`30` exposes peak motion alpha plus mean opacity/radius
  diagnostics in the sidebar, then bumps the asset cache and keeps the desktop
  rail scrollable. This is a measurement pass for the remaining convergence
  question: whether motion coverage falls because splats shrink/fade, or
  because the simplified representation is capacity-limited.
- `converge31` adds a small motion-coverage hinge to the simplified WGSL
  backward. On motion-sampled pixels below 50% dynamic alpha coverage, an extra
  weight-0.20 alpha-gradient term flows into opacity, center, radius, motion,
  and temporal-center updates. RGB/color gradients remain driven by the
  source-view reconstruction loss. The first browser trace shows this setting
  is too strong: it preserves coverage but slows motion-loss improvement.
- `converge32` weakens that hinge to a late support guard: target `0.44`,
  weight `0.08`.
- `converge33` renames the UI selector from `Shader Mode` to `Motion Model` and
  keeps World Tubes-style as the default after a same-guard comparison tied the
  Dynamic splats-style result.
- `converge34` keeps training math unchanged, makes target/render equal-width on
  desktop, and adds an RGB versus amplified motion-residual target view so visual
  convergence can be judged against the sparse moving pixels instead of dark RGB
  intuition.
- `converge38`/`39` add result-side diagnostics. `Result View` can show RGB, a
  bright dynamic layer, or alpha support. The support view proves the model is
  finding the moving person, but it also exposes broad background support. The
  first fix is `converge39`'s lower temporal gate floor (`sigma*0.70` to
  `sigma*0.30`).
- `converge40` adds `Static Cov`, a low-motion alpha penalty, and opacity
  decay, but the initial `0.055` decay is too aggressive and falls below the
  44% motion-support guard. `converge41` lowers decay to `0.025` and is the
  better sparsity trade; `converge42` keeps the v41 train constants and thins
  static-coverage validation to reduce readback overhead.
- `converge43` adds a dedicated low-motion sample buffer and an 8% static
  sample reserve. This addresses a real sampling bug in the v42 objective:
  static cleanup was only reached through the small uniform tail of the
  motion-heavy sampler.
- `converge44` exposes that reserve as `Static Mix`, so the browser can compare
  `0%` v42-style sampling against the default `8%` static reserve without
  source edits.
- 2026-07-07 math audit fixes:
  - the Neural3D preview is 512x256, so decoding to 96x96 distorted the target;
    the browser loader now preserves aspect and decodes the preview to 128x64
  - the preview is also a side-by-side camera preview, which mismatches the
    source-view image-space objective; `converge11` crops wide previews to the
    left source-view pane and decodes the default target to 128x128
  - the Gaussian metric and render projection are now target-aspect-aware
  - `posRadius.z` was previously unused; it is now the primitive temporal
    center with a soft temporal gate in both train and render
  - temporal support is now a UI hyperparameter; the default is `0.30`, with a
    0.14-0.32 range
  - training now applies the same 3-sigma Gaussian support cutoff that the
    render billboard draws, reducing invisible far-field tails
  - initialization now uses aspect-proportional grids and time-local color mixed
    with average color
  - the frame loop now survives mode resets/dispose-create races instead of
    silently stopping continuous training
  - `converge13` adds a mean static background and alpha-over-style dynamic
    residual splats, replacing the earlier positive-only additive residual that
    could not darken or occlude the background
  - `converge14` changes the train objective from orderless alpha coverage to
    fixed-order source-over compositing, matching the current render blend order
    and using the correct under-color/suffix-transmittance terms for each
    splat's opacity and geometry gradients
  - `converge16` exposes `Grid Loss` and `Motion Loss` readbacks from the
    current GPU params; the first reload showed `Grid Loss 0.000186` but
    `Motion Loss 0.044978`, explaining why the page could look non-convergent
    despite a small displayed sample/global loss
  - `converge17` adds the motion-sample buffer to the train bind group and biases
    samples toward moving pixels while keeping a uniform sample tail
  - `converge18` makes reset/create atomic from the app's point of view so
    ResizeObserver and the frame loop cannot render a half-initialized trainer,
    and changes the default LR slider value to `0.90`
  - `converge19` seeds a later-drawn subset of splats from the motion sample set,
    which trades a slightly higher initial grid loss for much better initial and
    short-run motion loss
  - `converge20` keeps the sparse grid metric but computes visible `Motion Loss`
    over up to 4096 packed motion samples, falling back to the old grid-weighted
    motion loss only when no motion samples exist
  - `converge21` adds the visible `Steps/s` readout and fixes the trainer
    fallback default learning rate to `0.90`
  - `converge22` changes the default splat count to 512 after the live
    capacity sweep showed it as the best quality/throughput point
  - `converge23` changes the default temporal support to `0.30` after the live
    sweep showed better true motion loss than `0.26`

Key files:

- `web/dynaworld_browser_trainer/index.html`
- `web/dynaworld_browser_trainer/app.js`
- `web/dynaworld_browser_trainer/dataset.js`
- `web/dynaworld_browser_trainer/trainerWebGpu.js`
- `web/dynaworld_browser_trainer/README.md`

Latest evidence:

- Syntax checks passed:
  `node --check web/dynaworld_browser_trainer/app.js`,
  `node --check web/dynaworld_browser_trainer/dataset.js`, and
  `node --check web/dynaworld_browser_trainer/trainerWebGpu.js`.
- Browser smoke served from repo root:
  `python3 -m http.server 8080`, then
  `http://127.0.0.1:8080/web/dynaworld_browser_trainer/`.
- 2026-07-07 convergence/mode smoke:
  - app loaded `./app.js?v=20260707-converge14`
  - loaded `Neural3D coffee_martini preview`
  - selected the Apple WebGPU adapter
  - cropped the side-by-side preview to `Neural3D coffee_martini preview
    (source-view crop)` with target canvas 128x128
  - showed 384 splats, temporal support `0.26`, 96 samples per step, and
    `Ready.`
  - World Tubes-style source-crop smoke reached step 151 with finite displayed
    loss about `0.00013`
  - Dynamic splats-style source-crop smoke reached step 192 with finite
    displayed loss about `0.00016`
  - 768-splat Dynamic splats-style smoke reset cleanly, reached step 98 with
    finite displayed loss about `0.00009`, and ran continuous training at about
    `8 fps`
  - final browser console check reported zero errors and zero warnings
- 2026-07-07 `converge17` validation/sampler smoke:
  - syntax checks passed again for `app.js`, `dataset.js`, and
    `trainerWebGpu.js`; `curl -I` returned `200 OK`
  - in-app reload loaded `./app.js?v=20260707-converge17`, dataset
    `Neural3D coffee_martini preview (source-view crop)`, GPU `apple`, and
    `2018` motion frame/pixel samples
  - initial 384-splat metrics: `Grid Loss 0.000186`, `Motion Loss 0.044978`
  - World Tubes-style 384-splat short run: step `191`, `Grid Loss 0.000183`,
    `Motion Loss 0.044205`, displayed sample loss about `0.00694`, no warnings
  - Dynamic splats-style 384-splat short run: step `367`,
    `Grid Loss 0.000181`, `Motion Loss 0.043216`, displayed sample loss about
    `0.00709`, no warnings
  - Dynamic splats-style 768-splat short run: step `128`,
    `Grid Loss 0.000177`, `Motion Loss 0.043898`, displayed sample loss about
    `0.00689`, about `8 fps`, no warnings
- 2026-07-07 hyperparameter sweep and `converge18` smoke:
  - short dynamic-splats sweep at 384 splats / temporal `0.26` / 96 samples:
    LR `0.45` reached step `165` and improved motion loss by `0.000443`; LR
    `0.90` reached step `190` and improved motion loss by `0.001885`
  - LR `0.90` with narrower temporal support `0.18` improved motion loss by
    `0.001667`, so `0.26` remains the default support
  - the sweep exposed a reset-time console error where ResizeObserver could call
    `renderOnce()` while the new trainer existed but had not finished
    `init()`; `converge18` fixes this with local `nextTrainer` construction and
    an initialized-trainer render guard
  - in-app reload loaded `./app.js?v=20260707-converge18`, default LR `0.90`,
    dataset `Neural3D coffee_martini preview (source-view crop)`, GPU `apple`,
    and `2018` motion frame/pixel samples
  - Dynamic splats-style `converge18` short run: step `221`,
    `Grid Loss 0.000180`, `Motion Loss 0.042676`, motion improvement
    `0.002287`, no new console warnings/errors
  - World Tubes-style `converge18` short run: step `178`,
    `Grid Loss 0.000180`, `Motion Loss 0.042843`, motion improvement
    `0.002135`, no new console warnings/errors
- 2026-07-07 `converge19` motion-aware init smoke:
  - syntax checks passed again for `app.js`, `dataset.js`, and
    `trainerWebGpu.js`; `curl -I` returned `200 OK`
  - in-app reload loaded `./app.js?v=20260707-converge19` and
    `styles.css?v=20260707-converge19`
  - initial World Tubes-style metrics: `Grid Loss 0.000219`,
    `Motion Loss 0.037862`; initial Dynamic splats-style metrics:
    `Grid Loss 0.000219`, `Motion Loss 0.037912`
  - Dynamic splats-style short run: step `341`, `Grid Loss 0.000157`,
    `Motion Loss 0.027288`, motion improvement `0.010624`, no new console
    warnings/errors
  - World Tubes-style short run: step `143`, `Grid Loss 0.000165`,
    `Motion Loss 0.027732`, motion improvement `0.010130`, no new console
    warnings/errors
- 2026-07-07 longer `converge19` stability/capacity checks:
  - Dynamic splats-style 384-splat longer run stayed stable: step `1126`,
    `Grid Loss 0.000146`, old grid-weighted `Motion Loss 0.025342`, no new
    console warnings/errors
  - Dynamic splats-style 768-splat run improved quality slightly per step but
    remained much slower: step `321`, old grid-weighted `Motion Loss 0.024954`,
    about `6-10 fps`, no new console warnings/errors
  - sample-count sweep: 32 samples was noisy (`Motion Loss` bounced to
    `0.028316` after a forced step); 64 samples got close faster but was still
    noisier on longer runs (`0.026485` final). Keep 96 samples as the default.
- 2026-07-07 `converge20` true motion-sample metric smoke:
  - syntax checks passed again for `app.js`, `dataset.js`, and
    `trainerWebGpu.js`; `curl -I` returned `200 OK`
  - Dynamic splats-style 384-splat run loaded
    `./app.js?v=20260707-converge20`, started at true `Motion Loss 0.009779`,
    and reached step `520` with `Grid Loss 0.000153`, `Motion Loss 0.007004`,
    no new console warnings/errors
  - World Tubes-style 384-splat run started at true `Motion Loss 0.009814` and
    reached step `304` with `Grid Loss 0.000160`, `Motion Loss 0.007175`, no new
    console warnings/errors
- 2026-07-07 `converge21` throughput/readout smoke:
  - in-app reload loaded
    `app.js?v=20260707-converge21` and `styles.css?v=20260707-converge21`
  - Dynamic splats-style 384 splats on the Apple WebGPU adapter reached step
    `158` while running with `Steps/s 16.7`, `Grid Loss 0.000164`,
    true `Motion Loss 0.007612`, LR `0.9`, and 96 samples/step
  - after pause the stat settled to `Steps/s 0.0`; no new console warnings or
    errors were reported
- 2026-07-07 capacity/default sweep:
  - 384 splats, Dynamic splats-style, LR `0.9`, 96 samples/step:
    true `Motion Loss 0.009779 -> 0.006904` by step `648`, with `Steps/s`
    roughly `16.7-20.7`
  - 768 splats, same config: true `Motion Loss 0.009557 -> 0.006711` by step
    `339`, but `Steps/s` was only about `7.0-7.9`
  - 512 splats, same config: true `Motion Loss 0.009794 -> 0.006685` by step
    `553`, with `Steps/s` about `11.6-13.5`; this is the best current default
    tradeoff and becomes `converge22`
- 2026-07-07 `converge22` post-edit smoke:
  - in-app reload loaded
    `app.js?v=20260707-converge22` and `styles.css?v=20260707-converge22`
  - default World Tubes-style boot showed 512 splats, `Grid Loss 0.000208`,
    true `Motion Loss 0.009774`, LR `0.9`, 96 samples/step, and `2018` motion
    samples
  - Dynamic splats-style reset showed 512 splats, `Grid Loss 0.000206`, true
    `Motion Loss 0.009794`; the short run reached step `174` with
    `Grid Loss 0.000148`, true `Motion Loss 0.007406`, `Steps/s 12.0`, and no
    new console warnings/errors
- 2026-07-07 temporal-support sweep at 512 splats:
  - `0.18` reached true `Motion Loss 0.007093` by step `390`
  - `0.22` reached true `Motion Loss 0.006991` by step `385`
  - matched `0.26` reached true `Motion Loss 0.006877` by step `392`
  - `0.30` reached true `Motion Loss 0.006794` by step `375`, then
    `0.006591` by step `574`; no new console warnings/errors
  - decision: promote `0.30` to the default as `converge23`
- 2026-07-07 `converge23` post-edit smoke:
  - in-app reload loaded
    `app.js?v=20260707-converge23` and `styles.css?v=20260707-converge23`
  - default World Tubes-style boot showed 512 splats, temporal `0.30`,
    `Grid Loss 0.000216`, true `Motion Loss 0.009968`, LR `0.9`, 96
    samples/step, and `2018` motion samples
  - Dynamic splats-style reset showed 512 splats, temporal `0.30`,
    `Grid Loss 0.000214`, true `Motion Loss 0.009983`
  - short Dynamic splats-style run reached step `237`, `Grid Loss 0.000145`,
    true `Motion Loss 0.007114`, `Steps/s 12.6` while running, and no new
    console warnings/errors
- 2026-07-07 post-`converge23` sampler/LR probes at 512 splats, temporal
  `0.30`, Dynamic splats-style:
  - constant LR `0.90`, 96 samples/step, 75% motion samples: true
    `Motion Loss 0.009983 -> 0.006431` by about step `810`
  - dropping LR to `0.45` around step `360` was negative: it reached only
    true `Motion Loss 0.006523` by step `835`
  - 128 samples/step was also not worth promoting: true
    `Motion Loss 0.006481` by step `713` with lower throughput than 96
    samples/step
- 2026-07-07 `converge24`/`converge25` motion-mix sweep:
  - `converge24` adds a `Motion Mix` slider and forwards it to the WGSL train
    config as `motionSamplePermil`
  - 75% motion mix reached true `Motion Loss 0.006756` by step `436`
  - 90% motion mix reached true `Motion Loss 0.006623` by step `451`
  - 95% motion mix reached true `Motion Loss 0.006608` by step `444`, and a
    longer 95% run reached step `831`, `Grid Loss 0.000135`, true
    `Motion Loss 0.006320`, with no browser warnings/errors
  - decision: promote the default Motion Mix to 95% as `converge25`; keep
    512 splats, temporal support `0.30`, LR `0.90`, and 96 samples/step as the
    current browser defaults
  - post-edit `converge25` reload loaded
    `app.js?v=20260707-converge25` and `styles.css?v=20260707-converge25`,
    booted with Motion Mix `95%`, then Dynamic splats-style reached step `263`,
    `Grid Loss 0.000144`, true `Motion Loss 0.006886`, with no browser
    warnings/errors
- 2026-07-07 `converge26` capacity retest under 95% motion sampling:
  - 768 splats, Dynamic splats-style, temporal `0.30`, LR `0.90`, 96
    samples/step, Motion Mix `95%`: true `Motion Loss 0.009754 -> 0.006201`
    by step `522`, `Grid Loss 0.000135`, with observed running `Steps/s`
    mostly around `10-11`, and no browser warnings/errors
  - matched 512-splat rerun under the same settings: true
    `Motion Loss 0.009983 -> 0.006505` by step `565`, `Grid Loss 0.000139`,
    with no browser warnings/errors
  - decision: promote 768 splats as `converge26`; keep the upper slider bound
    at 768 until a 1024+ capacity/speed probe exists
  - 64 samples/step at the new 768-splat default is neutral, not a promotion:
    it reached true `Motion Loss 0.006218` by step `568` with no browser
    warnings/errors, close to but not clearly better than 96 samples/step at
    `0.006201` by step `522`; keep 96 samples/step as the default
  - post-edit `converge26` reload loaded
    `app.js?v=20260707-converge26` and `styles.css?v=20260707-converge26`,
    booted with 768 splats / Motion Mix `95%`, then Dynamic splats-style
    reached step `192`, `Grid Loss 0.000142`, true `Motion Loss 0.006806`,
    with no browser warnings/errors
- 2026-07-07 `converge27` diagnostics and `converge28` motion-init fix:
  - `converge27` adds `Motion Cov` and `Active` readbacks. Initial Dynamic
    splats-style state: `455/768` active splats, motion coverage `39.8%`,
    true `Motion Loss 0.009754`
  - `converge27` diagnostic run reached step `415`, `Grid Loss 0.000136`,
    true `Motion Loss 0.006318`, but motion coverage fell to `29.9%`; the
    model was improving the loss while becoming less dynamically covering
  - `converge28` increases motion-seeded splats from 38% to 48%, broadens their
    initial radii, and raises their initial opacity. Initial Dynamic
    splats-style state: `504/768` active splats, motion coverage `63.3%`, true
    `Motion Loss 0.011483`
  - `converge28` short run reached step `470`, `Grid Loss 0.000160`, true
    `Motion Loss 0.005853`, motion coverage `41.6%`, no browser warnings/errors
  - extended `converge28` run reached step `854`, `Grid Loss 0.000149`, true
    `Motion Loss 0.005459`, motion coverage `38.2%`, no browser
    warnings/errors
  - decision: keep `converge28`; it is a better motion-region default despite a
    small grid-loss tradeoff
- 2026-07-07 `converge29`/`30` diagnostics:
  - surfaced `Peak Alpha`, `Mean Opac`, and `Mean Radius` from the existing
    validation readback so the next in-browser convergence trace can separate
    under-coverage, shrinking radii, and fading opacity
  - bumped the browser asset version again after adding independent desktop
    rail scrolling; no training math changed
- 2026-07-07 `converge31` motion-support guard:
  - added a motion-sample-only coverage hinge in the train shader, target
    `0.50`, weight `0.20`
  - the guard only activates when dynamic alpha coverage is below target; it is
    not a full renderer/tape/shared-backward port
- 2026-07-07 `converge32` weakened motion-support guard:
  - `converge31` Dynamic splats-style trace reached step `468`,
    `Grid Loss 0.000244`, true `Motion Loss 0.006782`, motion coverage `53.7%`,
    mean opacity `8.5%`, mean radius `0.0147`, and no browser warnings/errors;
    this preserved support but lagged the previous `converge28` step-470 motion
    loss (`0.005853`)
  - weakened the guard to target `0.44`, weight `0.08`
  - `converge32` Dynamic splats-style trace reached step `169`, `Grid Loss
    0.000200`, true `Motion Loss 0.007038`, motion coverage `52.5%`, mean
    radius `0.0142`, no browser warnings/errors
  - continued trace reached step `445`, `Grid Loss 0.000189`, true
    `Motion Loss 0.006263`, motion coverage `48.4%`, mean radius `0.0141`
  - extended trace reached step `861`, `Grid Loss 0.000189`, true
    `Motion Loss 0.005914`, motion coverage `47.0%`, mean radius `0.0140`,
    no browser warnings/errors
  - read: `converge32` is not a raw-MSE win over `converge28` (`0.005459` at
    step `854`), but it keeps substantially more dynamic support (`47.0%` vs
    `38.2%`) while reaching a comparable motion-loss band; keep it as the
    current support-health default unless the next visual read shows the broader
    support is hurting perceived fit
- 2026-07-07 `converge33` mode-label and default check:
  - World Tubes-style under the same weakened guard reached step `295`,
    `Grid Loss 0.000196`, true `Motion Loss 0.006440`, motion coverage `49.8%`,
    mean radius `0.0141`, no browser warnings/errors
  - extended World Tubes-style reached step `629`, `Grid Loss 0.000194`, true
    `Motion Loss 0.005938`, motion coverage `47.4%`, mean radius `0.0140`, no
    browser warnings/errors
  - read: do not flip the default to Dynamic splats-style; World Tubes-style is
    effectively tied under the support guard and is the more project-aligned
    default
  - renamed the UI selector to `Motion Model`, because these are branches inside
    one simplified WGSL trainer rather than two full native shader-family ports
- 2026-07-07 `converge34` visual diagnostic pass:
  - added `Target View` with `RGB target` and `Motion residual`; the residual
    view displays amplified `abs(frame - mean_background)` and is diagnostic
    only
  - changed the desktop workbench to equal-width target/render columns, fixing
    the misleading tiny-target versus giant-render comparison
  - live reload confirmed `app.js?v=20260707-converge34` and
    `styles.css?v=20260707-converge34`; default boot remained World Tubes-style,
    `504/768` active splats, `Motion Loss 0.011522`, `Motion Cov 63.1%`, and
    `Motion Px 2018`
  - post-edit World Tubes-style trace reached step `268`, `Grid Loss 0.000193`,
    true `Motion Loss 0.006505`, motion coverage `50.2%`, peak alpha `9.2%`,
    mean opacity `8.5%`, mean radius `0.0141`, and no browser warnings/errors
- one longer automation wait timed out at the CDP layer while the app kept
  training; recovery read paused the run cleanly, so this is not evidence of a
  WebGPU/runtime failure
- 2026-07-07 `converge38`/`39` result-side diagnostics and temporal-floor fix:
  - added `Result View` modes: `RGB result`, `Dynamic layer`, and `Alpha support`
  - a first fragment-storage-buffer residual attempt blacked out the WebGPU render
    pane while the trainer still reported `Ready`; backing out the fragment
    background-buffer read and removing the extra fragment-position argument
    restored RGB rendering in `converge38`
  - `converge38` alpha support at step `68` showed strong coverage on the moving
    person but also a broad speckled dynamic layer across the background
  - diagnosis: the default temporal support `0.30` previously implied a temporal
    gate floor of `sigma*0.70 = 0.21`, so every splat kept about 21% temporal
    opacity in every frame
  - `converge39` lowers the floor to `sigma*0.30` clamped to `0.035..0.12`
    (`0.09` at the default)
  - boot metrics improved from `converge38` `Motion Loss 0.011522`, `Motion Cov
    63.1%` to `converge39` `Motion Loss 0.011099`, `Motion Cov 59.7%`
  - short `converge39` World Tubes-style trace reached step `72`, `Grid Loss
    0.000214`, true `Motion Loss 0.007788`, motion coverage `53.9%`, peak alpha
    `9.7%`, mean opacity `8.5%`, mean radius `0.0144`
  - support screenshot pixel check moved only slightly (`frac>30` in the result
    crop `27.1% -> 26.2%`), so this is a small improvement rather than a full
    sparsity fix
  - read: remaining convergence issue is not missing support; it is broad
    dynamic-layer support plus imperfect residual/color isolation
- 2026-07-07 `converge40`/`41`/`42` sparsity objective pass:
  - added `Static Cov`, measured as dynamic alpha on low-motion validation
    samples, plus a low-motion alpha penalty and global opacity decay in the
    WGSL train kernel
  - `converge40` used opacity decay `0.055`; boot stayed at `Motion Loss
    0.011099`, `Motion Cov 59.7%`, `Static Cov 2.8%`, but by step `239` it
    reached `Motion Loss 0.007012`, `Motion Cov 42.4%`, `Static Cov 2.4%`,
    `Active 368/768`, `Mean Opac 6.5%`, falling below the motion-support guard
  - `converge41` lowered opacity decay to `0.025`; by step `294` it reached
    `Motion Loss 0.006751`, `Motion Cov 44.6%`, `Static Cov 2.6%`,
    `Active 406/768`, `Mean Opac 7.3%`, `Grid Loss 0.000182`, and
    `Peak Alpha 7.8%`
  - screenshot crop comparison of alpha support showed v41 reduced broad
    background activity versus v39 (`frac>30 0.2618 -> 0.2438`, `frac>80
    0.0538 -> 0.0436`, `frac>140 0.0203 -> 0.0164`) without dropping below the
    support guard
  - `converge42` keeps v41 train constants and thins `Static Cov` validation to
    one quarter of low-motion grid samples; reload verified
    `app.js?v=20260707-converge42` and `styles.css?v=20260707-converge42`, boot
    metrics were `Motion Loss 0.011099`, `Motion Cov 59.7%`, `Static Cov 2.8%`,
    and a one-step smoke reached step `1` with `Motion Loss 0.010983`,
    `Static Cov 2.8%`, `Active 503/768`
  - artifacts:
    `output/browser_trainer/converge39_alpha_support_result_step72.png` and
    `output/browser_trainer/converge41_alpha_support_result_step294.png`
  - read: v41/v42 are a better sparsity default than v40, but not a complete
    visual convergence fix; compare longer v42 traces against v32/v33 before
    spending more time on support knobs
- 2026-07-07 `converge43` dedicated static-sample reserve:
  - added `computeStaticSamples(...)` in the browser dataset loader using the
    same `0.00045` low-motion energy threshold as the static alpha penalty
  - added a second WGSL sample buffer at train binding `7`; the 96-byte train
    config reuses the old pad slots for `staticSampleCount` and
    `staticSamplePermil`
  - default train config now reserves 8% of samples for low-motion pixels and
    clamps the motion sample rate to leave room for that reserve, so the default
    UI motion-heavy run is no longer relying on random uniform hits for static
    cleanup
  - in-app reload verified `app.js?v=20260707-converge43`,
    `styles.css?v=20260707-converge43`, `Motion Px 2018`, `Static Px 16384`,
    boot `Motion Loss 0.011099`, and boot `Static Cov 2.8%`
  - first v43 trace: step `153`, `Motion Loss 0.007153`, `Motion Cov 48.5%`,
    `Static Cov 2.7%`, `Active 459/768`, `Mean Opac 7.8%`
  - extended v43 trace: step `184`, `Motion Loss 0.007054`, `Motion Cov
    47.6%`, `Static Cov 2.6%`, `Active 448/768`, `Mean Opac 7.7%`
  - paused v43 trace: step `259`, `Grid Loss 0.000179`, `Motion Loss
    0.006803`, `Motion Cov 45.5%`, `Static Cov 2.6%`, `Peak Alpha 8.0%`,
    `Active 420/768`, `Mean Opac 7.4%`, `Mean Radius 0.0142`
  - browser console check returned no warnings/errors
  - read: v43 makes the static cleanup objective mathematically less accidental
    and is at least comparable to v41 at a similar support state, but it is not
    a visual-quality proof; next compare v43 against v32/v33 by saved alpha/RGB
    screenshots and then move to renderer/init parity
- 2026-07-07 `converge44` static-reserve A/B control:
  - added `Static Mix` slider, default `0.08`, range `0..0.16`
  - app now passes `staticSampleRate` into `trainer.trainStep(...)`; the trainer
    keeps `0.08` as the API default but no longer hardcodes it at the call site
  - the `Motion Mix` value now shows the effective motion share after the static
    reserve is applied; default requested `0.95` plus static `0.08` displays
    effective `92%`
  - setting `Static Mix` to `0%` in the in-app browser changed labels to
    `Motion Mix 95%`, `Static Mix 0%`, which recovers the v42-style sampler
    without editing files; restored to default `8%` afterward
  - in-app reload verified `app.js?v=20260707-converge44`,
    `styles.css?v=20260707-converge44`, `Motion Px 2018`, `Static Px 16384`,
    `Motion Mix 92%`, `Static Mix 8%`
  - one-step v44 WebGPU smoke reached step `1`, `Motion Loss 0.010985`,
    `Motion Cov 59.6%`, `Static Cov 2.8%`, `Active 503/768`; browser logs were
    empty
  - matched v44 control: `Static Mix 0%` reached step `274`, `Grid Loss
    0.000182`, true `Motion Loss 0.006794`, `Motion Cov 45.0%`, `Static Cov
    2.6%`, `Peak Alpha 7.9%`, `Active 414/768`, and `Mean Opac 7.4%`
  - matched default v44 control: `Static Mix 8%` reached step `271`,
    `Grid Loss 0.000183`, true `Motion Loss 0.006822`, `Motion Cov 45.3%`,
    `Static Cov 2.6%`, `Peak Alpha 7.9%`, `Active 415/768`, and
    `Mean Opac 7.4%`
  - read: static reserve is not the core convergence issue. Both arms settle
    near the hidden 44% support target.
- 2026-07-07 `converge45` support-guard exposure:
  - added `Support Guard` slider, default `0.52`, range `0.40..0.60`
  - `app.js` forwards `motionCoverageTarget` into `trainer.trainStep(...)`;
    `trainerWebGpu.js` uses that value instead of hardcoding `0.44`
  - in-app reload verified `app.js?v=20260707-converge45`, `Motion Mix 92%`,
    `Static Mix 8%`, `Support Guard 52%`, `Motion Px 2018`, and
    `Static Px 16384`
  - v45 trace reached step `297`, `Grid Loss 0.000200`, true
    `Motion Loss 0.007060`, `Motion Cov 48.2%`, `Static Cov 2.7%`,
    `Peak Alpha 8.1%`, `Active 406/768`, `Mean Opac 7.3%`, and
    `Mean Radius 0.0146`
  - read: raising the weak support guard preserves a few more points of motion
    coverage but costs MSE. The browser is still missing renderer/init parity;
    do not turn this into another local support-target sweep.
- 2026-07-07 `converge46` motion-centroid init:
  - added frame-level motion centroid velocity estimates from
    target-vs-mean-background residuals
  - motion-seeded splats initialize `motion.xy` from that image-space velocity
    and back-solve `posRadius.xy` so the initialized tube center still lands on
    the selected high-motion frame/pixel
  - in-app reload verified `app.js?v=20260707-converge46`, `Motion Mix 92%`,
    `Static Mix 8%`, `Support Guard 52%`, `Motion Px 2018`, and
    `Static Px 16384`
  - boot changed from v45's `Motion Cov 59.7%` neighborhood to `Motion Cov
    57.1%`, while boot `Motion Loss` stayed effectively flat at `0.011103`
  - v46 trace reached step `290`, `Grid Loss 0.000211`, true
    `Motion Loss 0.007036`, `Motion Cov 47.0%`, `Static Cov 2.8%`,
    `Peak Alpha 8.2%`, `Active 407/768`, `Mean Opac 7.4%`, and
    `Mean Radius 0.0146`
  - saved metrics:
    `outputs/browser_trainer/2026-07-07_v46_motion_init/motion_centroid_init_support52_step290_metrics.json`
  - read: the motion prior is a small fit win versus v45
    (`0.007036` vs `0.007060`) but a support loss (`47.0%` vs `48.2%`). Keep
    the mechanism as a useful initializer direction, but do not call it
    convergence-fixed.
- 2026-07-07 `converge47` local residual-match motion init:
  - replaced global-only frame centroid velocity for motion-seeded splats with
    a local residual-pixel match in adjacent frames, blended `75%` local /
    `25%` frame-centroid fallback
  - the initializer searches a `7px` window around each selected motion sample,
    scores residual color similarity plus a small spatial cost and energy
    reward, then back-solves the base center so the initialized tube still lands
    on the chosen frame/pixel
  - in-app reload verified `app.js?v=20260707-converge47`, `Motion Mix 92%`,
    `Static Mix 8%`, `Support Guard 52%`, `Motion Px 2018`, and
    `Static Px 16384`
  - boot improved versus v46: `Motion Loss 0.010755` and `Motion Cov 59.3%`
    versus v46 boot `0.011103` and `57.1%`
  - v47 trace reached step `279`, `Grid Loss 0.000194`, true
    `Motion Loss 0.006885`, `Motion Cov 48.1%`, `Static Cov 2.7%`,
    `Peak Alpha 8.2%`, `Active 414/768`, `Mean Opac 7.4%`, and
    `Mean Radius 0.0145`
  - saved metrics:
    `outputs/browser_trainer/2026-07-07_v47_local_motion_init/local_motion_init_support52_step279_metrics.json`
  - read: this is the first clean positive local browser result after v44:
    better fit than v45 (`0.006885` vs `0.007060`) at essentially matched
    support (`48.1%` vs `48.2%`), and better fit/support than v46's global
    centroid prior. It is still source-view image-space initialization rather
    than renderer/geometry parity.
- 2026-07-08 `converge48` looping preview and source/target camera strip:
  - added `Loop Time` and `Loop Speed` controls; preview time now advances by
    default in the render loop instead of staying pinned to the slider value
  - `loadPresetDataset()` now attaches source/target preview crops from the
    side-by-side Neural3D preview when the local video is available; the main
    training dataset remains the source-view crop
  - added a small camera strip under the target canvas showing the source and
    target crops at the same looped time as the WebGPU result
  - exact default training target is `128x128x8`: ffprobe reports the preview
    video as `512x256`, and the loader crops one `256x256` pane then decodes it
    with `maxLongEdge=128`
  - performance read: training cost in this browser prototype is dominated by
    splat count and samples-per-step because the compute path dispatches one
    worker per splat and `eval_model(...)` loops over all splats for each
    sampled pixel. Canvas/target resolution is comparatively cheap for training
    at the current fixed sample count, though render fragments and CPU
    validation still scale with pixels.
  - verification: `node --check` passed for `app.js`, `dataset.js`, and
    `trainerWebGpu.js`; `python3 -m http.server 8080` was restarted from repo
    root; `curl -I` returned `200 OK`; served HTML/assets contain
    `v=20260707-converge48`, `Loop Time`, `advancePreviewTime(...)`, and
    `previewViews`
  - limitation: the in-app browser connector reported zero attached tabs after
    the user interruption, so this pass has code/server verification but not a
    fresh in-app visual reload. Do not claim a visual browser smoke until the
    tab is visible to the connector again.
- 2026-07-08 `converge49` validation visual metrics and error map:
  - added sparse-grid visual quality metrics to validation readback:
    `Val MAE`, `Val PSNR`, and `Val SSIM`
  - `Val SSIM` is a global luma SSIM approximation over the same deterministic
    sparse validation grid as `Grid Loss`; it is not windowed MS-SSIM and is not
    used as a training loss
  - added `Target View -> Validation error`; this throttles GPU-param readback,
    evaluates the current source-view prediction on CPU for the selected frame,
    and draws a heat map of RGB error in the target pane
  - training math is unchanged: RGB reconstruction is still the data loss, with
    existing temporal gating, radius/opacity clamps, low-motion alpha penalty,
    opacity decay, static-sample reserve, and motion-support guard as the
    browser regularization stack
  - verification: `node --check` passed for `app.js`, `dataset.js`, and
    `trainerWebGpu.js`; `curl -I` returned `200 OK`
- Earlier screenshot artifact:
  `output/playwright/dynaworld_browser_trainer_smoke.png`.

Next useful experiment:

- Port the renderer side toward real STAR/WorldTubes parity: depth-aware
  tile/bin compositing, alpha/transmittance matching, source/heldout camera
  training data, and a saved browser-training bundle format. Do not promote
  this source-view prototype into `BASELINES.md`.

## WorldFoam Gate4 Shader Research

Status: **Active shader lane, not system parity**

Purpose: evaluate whether the WorldFoam/Gate4 fused-MSE paths can beat STAR
direct-atomic scaling under controlled local gates.

Current decisions:

- 12-site Gate4 fused-MSE gates are fast but not capacity/system parity.
- The paper-math lane is separate from this shader-speed ledger. It now has a
  polished appendix at `research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md`;
  the next math-facing gate is a cell-path optical-transfer fixture with
  same-representation replay and exact VJP finite differences. The code-level
  spec is
  `research_notes/worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md`.
- The first three-lane visual smoke now exists at
  `outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_report.md`.
  It runs/records dynamic WorldFoam/PowerFoam Metal, WorldTubes/STAR UVT Metal,
  and base dynamic 3DGS fast-mac Metal on the tiny local dog clip. WorldFoam,
  STAR, and dynamic 3DGS now all write explicit disk media; the dynamic direct
  media rerun is
  `outputs/visual_comparisons/2026-07-07_dynamic_gsplat_direct_disk_media_rerun.json`.
  The clean all-lane summary is
  `outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_clean_all_lanes.json`.
- The comparison harness now supports `--tier medium`, a 128px/16f tier. The
  first medium run is green at
  `outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_medium_128_report.md`;
  all three Metal-backed lanes wrote local visual artifacts. This is a visual
  scale-up gate, not a representation-quality ordering claim.
- The comparison harness also supports `--tier capacity`. The first capacity
  row is green at
  `outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_capacity_128_report.md`.
  It uses WorldFoam 2048 cells / 80 steps, STAR 2048 tubes / 60 steps, and
  dynamic 3DGS 4096 explicit Gaussians / 60 steps. The dynamic config must keep
  `max_fast_pairs <= 2048` for the current compiled fast-mac `v6_refined`
  binary unless rebuilt with a larger `GSP_FAST_CAP`.
- 24-site high-cap rows now exercise candidates beyond the old 128 cap.
- Shell-sort/local-tape/owner-update variants were negative or insufficient.
- Lean `owner-run-fused-mse-nomid` recompute is the current compute keeper:
  clean 16px/24-site `2/4/8/16f` scales `1.13x` total and `1.18x`
  backward over an `8x` frame-count increase, but its selected tape storage
  still scales `9.65x`.
- Packed endpoint owner-run delta storage is the next justified shader fork:
  the probe preserves owner/count parity, recovers run lengths exactly, scales
  storage `5.76x` over `8x` frames, and at `16f` uses `90,220` bytes
  (`0.49x` of current nomid CSR). It is RGB-MSE-safe first; depth parity still
  needs internal moments/cuts or a semantic change.
- `owner-run-delta-packed-recompute-fused-mse-nomid` is now wired through the
  train/eval harness and has a moving-ray parity regression against
  `owner-run-fused-mse-nomid` (`1.64e-7` loss diff, `4.10e-7` max site-gradient
  diff in the pre-test probe). This is correctness-green, not timing-promoted:
  clean ladder timing was blocked by `benchmark_environment.status=contended`.
- Coeff factorization is the next storage fork: a real 24-site `2/4/8/16f`
  probe shows boundary planes plus per-track linear ray coefficients use
  `30,096` f32 bytes versus `1,130,496` dense coeff16 bytes (`2.66%`), with
  zero dense-depth validity mismatches and `7.14e-5` max depth error versus the
  dense f32 formula.
- `owner-run-delta-packed-factorized-recompute-fused-mse-nomid` is now a real
  Metal path. It removes resident `delta_coeff_f16`, stores `boundary_f32 +
  track_ray_coeff_f32`, passes moving-ray loss/site-gradient parity against
  `owner-run-fused-mse-nomid`, and has a functional `2/4/8/16f` ladder at
  render16/site8 with all rows `status=ok`. That ladder is not a clean speed
  promotion because `benchmark_environment.status=contended`, but it shows
  selected schema storage scaling `1.875x` over an `8x` frame-count increase
  and resident coeff storage scaling `1.0x`.
- The same factorized Metal path now consumes int16 metadata for base offsets,
  track-change offsets, change-frame rows, and change offsets. The old int32
  metadata buffers are absent from the selected MPS tape. The render16/site4/2f
  smoke reports actual schema storage `35,946` bytes and actual non-coeff MPS
  resident storage `11,306` bytes; the previous projected `4,102` byte metadata
  saving has been realized.
- The focused local gate now includes a `2/4/8f` storage regression and a
  24-site high-cap regression for the factorized mode: it asserts no
  `delta_coeff_f16`, asserts `boundary_f32 + track_ray_coeff_f32` residency,
  checks constant coeff storage plus sublinear selected storage, and verifies
  the factorized high-cap resident coeff storage is below `5%` of the dense
  `delta_coeff_f16` packed path. The resident coeff byte accounting was fixed
  to count both `boundary_f32` and `track_ray_coeff_f32` for this mode. Train
  rows emit `train_selected_tape_schema_storage_by_key` and keep the
  int16-metadata projection fields as an eligibility/audit surface. The
  combined WorldFoam owner-run factorization / packed-delta gate passes after
  the actual int16 metadata wiring.
- A pure no-frame-table variant is not valid for the moving-camera owner-run
  tape: actual sparse track events skip frames (for example `4f` site8 rows
  can be `[2, 3]`, not `[1, 2, 3]`). I instead added
  `owner-run-delta-packed-factorized-frameselect-recompute-fused-mse-nomid`, a
  real Metal fork that removes `track_change_offsets_i16`,
  `track_chunk_change_offsets_i16`, and `change_frame_i16` from the resident
  path and replaces the sparse scan with `frame_change_index_i16`, one selected
  sparse-change index per `(track, frame>0)`. The new kernel passes moving-ray
  loss/site-gradient parity against `owner-run-fused-mse-nomid`; the 24-site/8f
  regression proves the old scan metadata is absent and selected schema storage
  is below regular factorized; the focused owner-run suite now passes 8 tests
  in 360.265s. A render16/site8/4f smoke wrote
  `/tmp/worldfoam_factorized_frameselect_smoke.json` with `status=ok`, schema
  storage `46,088` bytes, `frame_select_i16=3,072` bytes, and no
  `track_change_offsets_i16` / `change_frame_i16` schema keys. That smoke is
  timing-contended by unrelated Python/Metal compiler work, so it is
  correctness/storage evidence only.
- A later regular-factorized `2/4/8/16f` timing attempt started from a clean
  preflight but ended with `benchmark_environment.status=contended`; unrelated
  pytest, STAR UVT training, `ai_trader` Python, and Metal compiler work
  appeared before the end snapshot. That artifact is useful as a rejected
  timing sample only. I added
  `research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py`
  plus unit coverage so the regular-vs-frame-select comparison now has
  per-mode stable preflights, refuses contaminated artifacts by default, and
  writes per-frame total/backward/storage ratios plus a recommendation.
- A first site8 comparison attempt produced one clean regular artifact and one
  contaminated-but-functional frame-select artifact. Frame-select was
  directionally faster on that attempt (`2/4/8/16f` total medians
  `1.779/2.158/2.079/3.541 ms` versus regular
  `5.655/2.984/2.790/3.573 ms`), but it cannot be promoted because the
  frame-select end snapshot was contended. The 16f frame-select schema storage
  was also higher (`74,046` bytes versus regular `67,014`) because the dense
  `frame_change_index_i16` table overtakes the sparse scan metadata it
  removes. The comparison gate now supports `--max-comparison-attempts` to
  automatically retry contaminated artifacts with distinct attempt outputs.
- `owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid` is
  now a real Metal path. It replaces the dense frame-select table with
  `track_frame_mask_i32` plus `track_change_offsets_i16` and uses rank/popcount
  lookup in the shader. Moving-ray loss/site-gradient parity passes against
  `owner-run-fused-mse-nomid`; the storage regression proves there is no
  resident `frame_change_index_i16`, `change_frame_i16`,
  `track_chunk_change_offsets_i16`, or stale `track_change_offsets_i32`. The
  focused owner-run suite passes 9 tests in 366.467s. A render16/site8/16f CLI
  smoke wrote `/tmp/worldfoam_factorized_framebitmask_smoke.json` with
  `status=ok`, schema storage `61,760` bytes, topology storage `36,624` bytes,
  and non-coeff MPS resident storage `36,736` bytes. That smoke is
  timing-contended by a STAR UVT run, so this is correctness/storage evidence,
  not speed promotion.
- `compare_factorized_frameselect_gate.py` now supports `--include-framebitmask`
  and can run a regular/frame-select/frame-bitmask side-by-side ladder with
  per-mode stable preflights, contaminated-artifact retries, and aggregate
  candidate selection. Unit coverage verifies that frame-bitmask wins when
  frame-select regresses storage and that frame-bitmask contamination retries the
  full comparison attempt. A live short-window attempt wrote
  `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_blocked_live.factorized_frameselect_compare_summary.json`
  with `status=preflight_failed_before_regular`; no timing rows were launched.
  A longer clean-window run
  `2026-05-19_factorized_selector_compare_clean_site8` was interrupted for
  reflection before a usable comparison: attempts 1 and 2 produced only regular
  factorized artifacts and both ended contended, so the gate retried; attempt 3
  was still `waiting_for_preflight`. There are no frame-select or frame-bitmask
  timing artifacts for that run, and `comparison=null`. The comparison gate now
  catches Ctrl-C/SIGTERM during future waits/runs and writes
  `status=interrupted` instead of leaving a stale waiting summary; the focused
  comparison-gate unit suite passes 9 tests.
  A bounded retry
  `2026-05-19_factorized_selector_compare_clean_site8_retry_interruptfix`
  reached regular factorized after two clean preflights, but pytest and STAR UVT
  work appeared before the end snapshot. The regular artifact was contended and
  failed its internal sublinear acceptance, so no candidate rows were launched.
  That exposed and fixed a compare-gate bug: nonzero train/eval exits now load a
  written artifact and retry only when its benchmark environment is contaminated.
  A second retry
  `2026-05-19_factorized_selector_compare_clean_site8_retry2_nonzero_retryfix`
  proved the next edge case: attempt 1 had good regular-factorized scaling
  (`total_step_scale_first_to_last=1.317`, `backward_scale_first_to_last=1.378`)
  but was rejected for end-snapshot contamination, then attempt 2 hit child-side
  start-check contamination before writing `out_json`. The gate now treats child
  exit `2` without an artifact as retryable start-environment contamination and
  writes the current attempt status before hard returns. The same artifact also
  exposed stale top-level summary fields across attempts; the gate now clears
  per-mode result fields before each mode run so later start-check contamination
  cannot inherit old artifact metadata. The focused comparison-gate suite now
  passes 12 tests.
  A third retry
  `2026-05-19_factorized_selector_compare_clean_site8_retry3_stalefieldfix`
  was stopped for reflection while waiting for another clean preflight. It
  produced one clean regular-factorized artifact on attempt 2:
  `total_step_scale_first_to_last=1.744`,
  `backward_scale_first_to_last=1.263`, and 16f selected-tape storage at
  `3.75%` of full. The same attempt reached dense frame-select and the timing
  shape looked faster (`1.766/2.755/2.354/3.064 ms` total for `2/4/8/16f`),
	  with matching train/heldout PSNR, but an unrelated pytest job contaminated the
	  end snapshot. That artifact is not promotable, and frame-bitmask still has no
	  clean side-by-side speed artifact.
	- Follow-up gate work added accepted-clean-artifact reuse and explicit
	  `--candidate-labels`, then the frame-bitmask shader was tightened twice:
	  cached per-track mask/offset reads and parallel per-frame selector setup
	  replaced the old serial local-frame-0 selector fill. Validation:
	  comparison-gate py_compile, the focused comparison-gate unittest suite
	  (`15` tests), focused owner-run frame-bitmask parity/storage tests, rebuild of
	  `world_foam_lane2_fused_slab_v0`, and
	  `/tmp/worldfoam_framebitmask_parallel_setup_smoke.json` with `status=ok`.
	  The first clean post-cache retry
	  `2026-05-19_factorized_selector_compare_clean_site8_retry9_framebitmask_masksetup_quiet`
	  proved the old 4f failure was fixed but exposed an 8f timing spike
	  (`1.131x` total, `1.171x` backward), so it was not promotable. The final
	  clean retry
	  `2026-05-19_factorized_selector_compare_clean_site8_retry11_framebitmask_parallelsetup`
	  is promotable at site8: `status=ok`, `best_candidate=framebitmask`,
	  background env, max total ratio `0.884`, max backward ratio `0.886`, max
	  schema ratio `0.973`, max topology ratio `0.922`, and max non-coeff resident
	  ratio `0.923` versus the accepted clean regular artifact. Per-frame
	  total/backward ratios for `2/4/8/16f` are
	  `0.809/0.884/0.809/0.878` and `0.856/0.886/0.812/0.876`. This promotes
	  frame-bitmask as the current site8 WorldFoam selector winner, but not yet as
	  a site24/high-cap or STAR UVT competitor.
	- The site24/high-cap repeat also passes. The paired gate
	  `2026-05-19_factorized_selector_compare_clean_site24_retry1_framebitmask_parallelsetup`
	  ran regular factorized once cleanly, rejected the first frame-bitmask attempt
	  for an unrelated `ai_trader` end-snapshot spike, then accepted frame-bitmask
	  attempt 2 with background start/end snapshots. Summary: `status=ok`,
	  `best_candidate=framebitmask`, max total ratio `0.942`, max backward ratio
	  `0.941`, max schema ratio `0.978`, max topology ratio `0.940`, and max
	  non-coeff resident ratio `0.940`. Per-frame total/backward/schema ratios for
	  `2/4/8/16f` are `0.854/0.916/0.864/0.942`,
	  `0.869/0.941/0.856/0.927`, and `0.978/0.952/0.927/0.909`. This promotes
	  frame-bitmask for both site8 and site24/high-cap synthetic WorldFoam
	  selector comparisons; the remaining proof is the matched STAR UVT speed
	  comparison.
	- The first render64/site24 frame-bitmask follow-up exposed a real metadata
	  scale issue rather than a stale-buffer bug: `change_offsets_i16` overflowed
	  at 4f (`55,797`) and `track_change_offsets_i16` overflowed at 8f
	  (`51,154`). The frame-bitmask fork now uses
	  `track_change_offsets_i32 + change_offsets_i32` while regular factorized
	  framegroup16 remains on int16 metadata. Rebuild plus focused owner-run
	  tests pass. The clean 2/4f artifact
	  `2026-05-19_worldfoam_framebitmask_render64_site24_i32offsets_2_4_steps8_warm4`
	  is `status=ok` with `1.218x` total and `1.242x` backward for 2x frames.
	  Contended 8f/16f correctness artifacts are also `status=ok`, but not
	  speed-promotable. The 16f artifact shows the new practical bottleneck:
	  train prep spent `221.47s` in endpoint-record sequence build, `455.77s`
	  in segment-tape build, and `137.85s` in baseline-tape compaction before a
	  single GPU step.
	- Added `--experimental-selected-only-owner-run-delta-prep` for
	  `slow-owner-run` owner-run delta packed modes. This skips the full
	  segment-tape and baseline compaction phases for selected shader timing
	  artifacts while explicitly marking rows with
	  `train_baseline_segment_metrics_built=false`. The trainer path also disables
	  unused owner-run `sample_meta` allocation. The selected-only path derives
	  expanded semantic row counts from delta frame-row descriptors, so
	  `selected_segments` still matches the full owner-run metric without
	  rebuilding the baseline tape. Focused owner-run tests pass (`3` tests,
	  `74.745s`). The render64/site24/4f path smoke
	  `2026-05-20_worldfoam_framebitmask_render64_site24_selected_only_prep_4f_smoke`
	  is `status=ok` and contains no `build_segment_tape_s` or
	  `compact_baseline_tapes_s` timings; it is not speed-promotable because the
	  end benchmark snapshot was contended. The remaining prep bottleneck is now
	  isolated to `slow-owner-run` endpoint-record sequence construction
	  (`54.09s` train at 4f in the selected-only smoke; `221.47s` train at 16f in
	  the previous full-metric artifact).
	- Added `--experimental-native-owner-run-cutwalk-delta` as an exact C++ owner-run
	  delta builder from boundaries/sites/rays. The first approximate cutwalk
	  attempt was wrong because crossing any boundary involving the current owner
	  treats inactive power-cell boundaries as owner transitions; the fixed op
	  computes midpoint nearest-site owners per cut interval, merges same-owner
	  runs, and applies the same transmittance threshold as the Python reference.
	  The previously failing native framebitmask shader parity test now passes,
	  and the full owner-run delta packed suite passes (`8` tests, `561.792s`).
	  CPU native-cutwalk parity now also covers a duplicated multiview/moving-ray
	  fixture, so the C++ cutwalk builder is checked against Python owner-run
	  sequences for view-major sample order instead of only the original
	  single-view moving-ray case (`2` focused CPU tests, `28.622s`). The matching
	  MPS shader-output regression also passes for the original and duplicated
	  shifted multiview fixtures (`2` focused tests, `27.706s`). Path smoke
	  `2026-05-20_worldfoam_framebitmask_render64_site24_selected_only_native_cutwalk_4f_smoke`
	  is `status=ok`; train sequence prep dropped from the old selected-only
	  Python `54.09s` 4f row to `0.553s`. The `2/4/8/16f` path ladder
	  `2026-05-20_worldfoam_framebitmask_render64_site24_selected_only_native_cutwalk_2_4_8_16_path`
	  is also `status=ok` with environment status `background`, all acceptance
	  keys true, train sequence prep `0.190/0.693/1.261/1.329s`, and backward
	  medians `2.304/2.749/4.845/3.739ms`. Treat it as strong path/scaling
	  evidence, but still rerun with `--require-benchmark-environment-ok` before
	  using it as the clean STAR UVT competitiveness row; a `steps=8/warm4`
	  gated attempt correctly refused to start while an unrelated `ai_trader`
	  Python process was consuming about `95%` CPU, and a later preflight was
	  still blocked by unrelated font-experiment Python/pytest work. A later
	  `clean_retry2` artifact completed the `2/4/8/16f` rows but ended with
	  benchmark environment `contended`, so it is also diagnostic only:
	  backward medians `2.278/2.738/2.824/3.844ms`, total medians
	  `2.568/3.170/3.141/4.230ms`, and train prep
	  `0.117/0.342/0.683/1.846s`. `clean_retry3` started from a clean
	  preflight but also ended `contended` after unrelated `ai_trader` export
	  and font-training work appeared; it is diagnostic only with backward
	  medians `2.698/22.295/3.980/6.014ms`, total medians
	  `3.137/28.368/4.790/6.541ms`, and train prep
	  `0.248/0.823/1.227/2.803s`. The STAR comparison harness now has
	  `--require-clean-worldfoam-artifact` and
	  `--require-benchmark-environment-ok` so it cannot promote a contended
	  WorldFoam artifact or contended STAR timing run. Both the WorldFoam
	  ladder and STAR comparison also support
	  `--wait-for-benchmark-environment-ok-timeout-s`; a 30s live smoke of the
	  WorldFoam wait path correctly waited and then exited `2` because unrelated
	  dynamic-gsplat, `ai_trader`, font-training, and git-add work stayed
	  contending. Added
	  `run_worldfoam_star_native_cutwalk_gate.py` as the unattended promotion
	  wrapper: it waits for a clean WorldFoam preflight, runs the native-cutwalk
	  render64/site24 ladder, rejects non-promotable artifacts, then runs the
	  guarded 64px/896-tube STAR comparison. Dry-run and preflight-blocked unit
	  tests pass; the focused wait/wrapper/STAR comparison unit set passes `11`
	  tests. A short live blocked smoke wrote a promotion summary and did not
	  start STAR. A later bounded wait attempt
	  `2026-05-20_native_cutwalk_worldfoam_star_wait_attempt1` caught a clean
	  start preflight and completed WorldFoam rows, but the end snapshot became
	  `contended` after unrelated font-training and `ai_trader` jobs appeared,
	  so STAR was not launched. Treat it as diagnostic only: backward medians
	  `2.700/2.738/3.044/5.892ms`, total medians
	  `3.116/3.240/3.651/6.865ms`, and train prep
	  `0.295/0.592/1.199/2.452s`. The wrapper status label is now tightened so
	  an existing-but-contended WorldFoam artifact is reported as
	  `worldfoam_not_promotable` instead of generic failure. The wrapper also
	  supports `--max-worldfoam-attempts`, retrying preflight timeouts or
	  end-contended WorldFoam artifacts and passing the first clean WorldFoam
	  artifact to STAR. The wrapper now runs and records an explicit
	  `--benchmark-environment-check-only` preflight before each WorldFoam
	  attempt, so blocked attempts preserve the blocker snapshot instead of
	  ending as an opaque no-artifact failure. Live audit
	  `2026-05-20_native_cutwalk_worldfoam_star_preflight_audit` exited
	  `worldfoam_preflight_failed_or_contended` without launching WorldFoam and
	  recorded active `ai_trader` feature-context and pytest blockers. The
	  wrapper also has `--preflight-only` for an intentional readiness audit;
	  live audit `2026-05-20_native_cutwalk_worldfoam_star_preflight_only_audit`
	  exited without launching WorldFoam and recorded active `font_maker`,
	  `ai_trader` SFT, and STAR UVT feature-overfit blockers. Focused
	  wait/wrapper/STAR tests now pass `14` tests. The promotion summary now
	  includes compact `worldfoam_preflight_blocking_processes` alongside the
	  full environment snapshot; live
	  `2026-05-20_native_cutwalk_worldfoam_star_preflight_compact_recheck`
	  records `font_maker` training and pytest blockers without launching
	  WorldFoam. The compact summary now filters to actual high-CPU blockers
	  when present; live
	  `2026-05-20_native_cutwalk_worldfoam_star_preflight_filtered_continue`
	  records only the hot `font_maker` training and `ai_trader` export child in
	  the compact blocker list, while the full snapshot still retains idle parent
		  wrappers. Focused wait/wrapper/STAR tests now pass `15` tests. While timing
		  was blocked, the promotion wrapper was hardened so top-level
		  `worldfoam_artifact` now means the clean promotable artifact selected for
		  STAR. Failed preflights keep `worldfoam_artifact=null`, while diagnostics
		  live under `planned_worldfoam_artifact`,
		  `worldfoam_latest_attempt_artifact`, and
		  `worldfoam_latest_written_artifact`. Live post-change preflight-only audit
		  `2026-05-20_native_cutwalk_worldfoam_star_preflight_contract_resume` exited
		  `worldfoam_preflight_failed_or_contended` without launching WorldFoam and
		  showed the new `worldfoam_artifact=null` contract; hot blockers were
		  `font_maker` training (`~137%` CPU) and `ai_trader` SFT shadow
		  (`~18.5%`). A newer preflight-only check
		  `2026-05-20_native_cutwalk_worldfoam_star_preflight_now` was still
		  contended by `font_maker` training (`~120.7%` CPU), an `ai_trader`
		  tree-residual export child (`~86.4%`), and git add (`~35.4%`), so no
		  full timing run was launched. Added a regression test for the exact stale
		  path bug from the multi-attempt wrapper: a contended written artifact
		  followed by a later preflight failure must keep `worldfoam_artifact=null`,
		  retain `worldfoam_latest_written_artifact` at the written diagnostic, and
		  move `worldfoam_latest_attempt_artifact` to the unwritten retry. The STAR
		  command field is now split too: `planned_star_compare_command` records the
		  planned audit command, while selected `star_compare_command` remains null
		  until a clean WorldFoam artifact is actually passed to STAR. Live
		  `2026-05-20_native_cutwalk_worldfoam_star_preflight_starcmd_contract`
		  proved the new shape under contention with `star_compare_command=null`;
		  blockers were `ai_trader` SFT shadow (`~81.9%`) and a STAR dense-alpha
		  diagnostic (`~45.4%`). The focused wait/wrapper/STAR suite still passes
		  `16` tests. The wrapper summary now emits
		  `summary_schema_version=worldfoam_star_native_cutwalk_gate_v2` so older
		  pre-contract artifacts are distinguishable from the tightened shape. Live
		  audit `2026-05-20_native_cutwalk_worldfoam_star_preflight_schema_contract`
		  exited `worldfoam_preflight_failed_or_contended` without launching
		  WorldFoam or STAR, kept `worldfoam_artifact=null` and
		  `star_compare_command=null`, and recorded a hot `font_maker` torch child
		  (`~152.6%` CPU) while the `ai_trader` shadow monitor was visible but idle
		  in the background snapshot. `py_compile` and the focused
		  wait/wrapper/STAR suite passed after the schema marker change. A bounded
		  full wrapper attempt
		  `2026-05-20_native_cutwalk_worldfoam_star_clean_wait120` then waited
		  through three 120s preflight attempts and still exited
		  `worldfoam_preflight_failed_or_contended`, with no WorldFoam artifact
		  written and no STAR command selected. Attempt blockers stayed external:
		  `font_maker` torch remained hot (`~139-156%` CPU) and intermittent
		  `ai_trader` SFT children hit `~77-95%` CPU. Added a focused regression
		  for this all-preflight-failures path: three contended preflight attempts
		  must leave `worldfoam_latest_written_artifact=null`,
		  `worldfoam_artifact=null`, `star_compare_command=null`, and preserve the
		  final compact blockers. The focused wait/wrapper/STAR suite now passes
		  `17` tests. The final acceptance audit now lives at
		  `research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py`.
		  It requires a schema-v2 `status=ok` summary, one selected/promotable
		  WorldFoam artifact, clean WorldFoam and STAR benchmark environments, the
		  native-cutwalk/selected-only flags in the WorldFoam artifact, and STAR
		  consuming the selected artifact; verifier unit coverage passes `3` tests
		  and the current blocked preflight summary correctly fails verification.
		  The promotion wrapper now has `--verify-promotion`; when the guarded
		  STAR comparison succeeds, the wrapper writes the summary, runs the
		  verifier, records `promotion_verifier_*` fields, and returns
		  `promotion_verification_failed` if the audit rejects the summary. Live
		  preflight-only audit
		  `2026-05-20_native_cutwalk_worldfoam_star_preflight_verifier_integration_blocked`
		  exercised the verifier-integrated wrapper command shape but correctly
		  refused to launch WorldFoam or STAR because external `font_maker`
		  training and an `ai_trader` activation RL dataset build made the
		  benchmark environment contended. A later readiness check remained
		  contended by `font_maker` torch and an `ai_trader` pytest training
		  child, so no clean timing row was launched. Non-timing validation after
		  the verifier integration passes: wrapper/wait/STAR/verifier contract
		  suite `22` tests, native cutwalk CPU parity `2` tests in `25.409s`, and
		  selected framebitmask native-cutwalk MPS shader-output parity `2` tests
		  in `27.771s`. The wrapper now has an opt-in
		  `--preflight-stability-samples` guard so the clean gate can require
		  multiple consecutive clean preflight samples before launch; coverage is
		  now `23` wrapper/wait/STAR/verifier tests after adding a late-contention
		  regression. Live
		  `2026-05-20_native_cutwalk_stable_preflight_blocked` requested `3`
		  stability samples with `5s` spacing and correctly stopped at the first
		  contended sample, leaving `worldfoam_artifact=null` and
		  `star_compare_command=null`.
		  The follow-up strict-background gate is the first clean promotion row:
		  `2026-05-20_native_cutwalk_worldfoam_star_starretry.promotion_summary.json`
		  exits `status=ok`, selects one promotable WorldFoam artifact and one
		  promotable STAR artifact, requires `benchmark_environment.status=background`
		  for both, and passes the integrated promotion verifier with no failures.
			  The wrapper rejects missing benchmark environments, treats `ok` and
			  `background` as clean, supports `--max-star-attempts`, and retries a
			  contended STAR comparison without rerunning a clean WorldFoam artifact.
		  WorldFoam means are `3.008/3.014/3.323/4.095ms` total and
		  `2.739/2.517/2.561/3.796ms` backward for `2/4/8/16f`
		  (`1.361x` total scale, `1.386x` backward scale), with heldout PSNR
		  `12.352/12.406/12.589/12.857`. The matched STAR medians are
		  `5.003/5.943/8.092/9.794ms` total and `2.629/3.411/5.083/6.768ms`
		  backward (`1.957x` total scale, `2.574x` backward scale), so this is a
		  clean Gate4 speed/scale win, not a broad quality/system-parity claim.
		  The repeated-fixture 32f extension
		  `2026-05-20_native_cutwalk_worldfoam_star_repeat32_framebitmask32fix`
		  also passes the strict wrapper and verifier, after fixing the
		  framebitmask signed-int32 bit-31 boundary. It requests `2/4/8/16/32f`
		  with `--repeat-loaded-frames`; both WorldFoam and STAR explicitly record
		  that the 32f row repeats a 16-frame loaded target. WorldFoam medians are
		  `2.829/3.248/4.414/4.643/6.371ms` total and
		  `2.557/2.965/4.054/4.254/6.001ms` backward (`2.252x/2.347x` scale over
		  requested `2 -> 32f`), while matched STAR medians are
		  `5.324/6.436/7.623/9.937/13.344ms` total and
		  `2.770/3.495/4.474/6.126/9.013ms` backward (`2.506x/3.254x`). Treat
		  this only as a synthetic speed-scaling smoke; the next stronger gate
		  needs a real longer-than-16f fixture or a larger quality-linked setup.
		  A later render96/site48 functionality smoke exposed that the
		  framebitmask shader still treated `base_offsets_i32` as int16 metadata:
		  max base offset reached `83695`, so the path failed before writing an
		  artifact. The framebitmask fused-MSE shader now consumes int32 base
		  offsets, keeps only the still-valid compact frame mask, and records clean
		  storage accounting in
		  `2026-05-20_worldfoam_native_cutwalk_render96_site48_2f_functionality_smoke.json`.
		  This fixes the larger-record correctness blocker, not a promotable timing
		  point; the smoke environment was contended by unrelated ai_trader work.
		  The follow-up strict render96/site48 gate
		  `2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate`
		  then passed after the wrapper rejected a first attempt contaminated by a
		  live ai_trader Toto export. Attempt 2 is the promotable WorldFoam
		  artifact, benchmark environment `background`, integrated verifier
		  `ok`. WorldFoam medians are `3.760/4.125/4.619ms` total and
		  `3.480/3.847/4.331ms` backward for `2/4/8f` (`1.229x/1.245x` median
		  scale), with heldout PSNR `10.801/10.838/10.880`; matched STAR
		  96px/1792-tube medians are `5.773/7.583/9.692ms` total and
		  `3.614/5.161/6.719ms` backward (`1.679x/1.859x`). The gate proves the
		  i32-base-offset framebitmask path survives a larger render/site case and
		  remains faster than matched STAR in this fused-MSE micro-setup; it is
		  still not RGB-quality or system parity.
		  Follow-up harness work added a strict real-frame contract for the next
		  promotion attempt. `run_worldfoam_star_native_cutwalk_gate.py` can now
		  forward `--worldfoam-config` to the WorldFoam train/eval script and
		  `--star-video-path` to the STAR comparison, while
		  `--require-real-loaded-frames` makes
		  `verify_worldfoam_star_native_cutwalk_promotion.py` reject summaries or
		  artifacts that used repeated loaded frames, too-small
		  `loaded_frame_count`, or missing loaded-frame metadata. The fixture
		  inventory note is now refined: checked 16f manifests were the blocker,
		  but raw DeepView can build a reproducible real 32f heldout-multicam
		  fixture. Config
		  `src/dataset_configs/multicam_val_deepview_03dog_128_8fps_32f.jsonc`
		  builds
		  `data/multicam_val/clip_sets/multicam_val_deepview_03dog_128_8fps_32f/manifest.jsonl`,
		  and
		  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc`
		  points WorldFoam at two train cameras plus heldout `camera_0040`.
		  The one-step real32 native-cutwalk smoke
		  `2026-05-20_worldfoam_real32_native_cutwalk_loader_smoke.json` passes
		  as a data/correctness gate (`loaded_frame_count=32`,
		  `repeat_loaded_frames=false`, loss decreased, gradients and parameter
		  update nonzero), but ended with `benchmark_environment.status=contended`
		  from `MTLCompilerService`. Do not treat its `727ms` total /
		  `677ms` backward row as speed evidence. The dry-run wrapper summary
		  `2026-05-20_real32_dryrun.promotion_summary.json` verifies the strict
		  command shape with `--worldfoam-config`, `--star-video-path`, and
		  `--require-real-loaded-frames`; it has been refreshed after the
		  frame-count-contract hardening and now records `frame_counts=[32]`.
		  The next promotion needs a quiet full wrapper run. The warm real32 settle retry
		  `2026-05-20_real32_strict_mini_wrapper_settle_retry` found two clean
		  preflight gaps and proved the warm native-cutwalk shader step itself is
		  small (`2.25/2.30ms` total, `1.95/2.01ms` backward at true 32f), but
		  both attempts ended diagnostic because the live `ai_trader` TOTO export
		  launched before the post-run benchmark snapshot. Because this is a real
		  Python/MPS blocker, not an `MTLCompilerService`-only transient, the new
		  settle path correctly did not promote it and STAR was not launched. The
		  benchmark classifier now marks the periodic TOTO MPS-export monitor as
		  blocking even when its screen/login/python parents are idle, so a live
		  check-only preflight should stop the next run before it wastes a
		  WorldFoam attempt in that 30-second export cadence. Wrapper preflight
		  artifact
		  `2026-05-20_real32_preflight_toto_mps_blocker_check.promotion_summary.json`
		  proves that top-level behavior: status
		  `worldfoam_preflight_failed_or_contended`, no WorldFoam artifact, no
		  STAR attempt, and idle TOTO parents listed as blocking processes with
		  `block_reason=periodic_mps_exporter` even when another live TOTO child
		  is the current high-CPU blocker. Follow-up verifier hardening closes
		  the last fake-real promotion escape: when
		  `require_real_loaded_frames=true`, the verifier now also requires
		  recorded `worldfoam_config` and `star_video_path`, plus matching
		  `--config` in the WorldFoam preflight/run commands, matching
		  `--video-path` in the planned/selected STAR commands, and
		  `--worldfoam-artifact` in the planned/selected STAR commands pointing
		  at the selected WorldFoam artifact. It also cross-checks the selected
		  artifact lineage: WorldFoam `config_path` must match
		  `worldfoam_config`, and STAR `star.video_path` must match
		  `star_video_path`. The wrapper now records parsed `frame_counts` in
		  the promotion summary, and real-frame verification requires the
		  WorldFoam rows and STAR rows to match that exact requested frame set.
		  The wrapper now also fails at argument parsing if
		  `--require-real-loaded-frames` is used without both
		  `--worldfoam-config` and `--star-video-path`, so a fake-real run cannot
		  burn a timing slot and be rejected only at final verification time. The
		  tests cover neither-input and one-sided-input failures. It also rejects
		  empty, non-integer, nonpositive, and duplicate `--frame-counts` at parse
		  time. The focused WorldFoam lane unit gate is now `56` tests passing,
		  with scoped `py_compile`, trailing-whitespace, and `git diff --check`
		  clean. The native owner-run cutwalk tests now also cover a synthetic
		  non-repeated `32f` moving-ray boundary: CPU cutwalk delta parity against
		  Python owner-run sequences and MPS framebitmask fused-shader output parity
		  both pass, so the 32-frame framebitmask path is no longer covered only by
		  a scalar bitmask unit and 4f shader parity. A direct low-level MPS
		  regression now also forces `track_frame_mask_i32 = -(1 << 31)` on a
		  one-track 32f tape and verifies the frame-31 change alters loss/grad
		  against the all-base tape, closing the signed-bit shader coverage gap
		  that the ordinary moving-ray fixture missed. The low-level MPS wrapper
		  now also rejects framebitmask tapes whose per-track mask popcount does
		  not match the per-track change-record span; the owner-run delta packed
		  module passed `17` tests after that guard. The CPU tape builder now
		  also rejects unsorted per-track change frames because the framebitmask
		  shader maps a frame bit to a change row by `popcount(lower_mask)`, so
			  change records must be strictly ascending by frame id. The
			  framebitmask helper now also rejects malformed change-offset vectors
			  directly: empty offsets, nonzero first offsets, nonmonotonic offsets,
			  and final offsets that do not match `change_frame_i32` length. The MPS
			  wrapper now has direct negative coverage for illegal mask bits too:
			  frame `0` and bit `frame_count` are rejected with popcount held
			  constant, so the tests exercise the bounds guard rather than the
			  popcount guard. The same sparse-change validation is now shared by the
			  frame-select helper, which rejects unsorted per-track frames, frame-0
			  changes, and non-1D offset tensors before building the int16 rank map.
			  The framebitmask MPS wrapper now also rejects empty
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
			  Handcrafted direct-config dictionaries and stale markers after record,
			  topology, config, or selector replacement/in-place mutation fail before
			  the native Metal launch instead of bypassing wrapper validation. Prepared
			  tapes set that marker after CPU packed-record range validation and native
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
					  wrapper/verifier gate now passes `72` tests after the guard refresh,
					  including `unchecked` benchmark-environment probes blocking strict
					  promotion instead of behaving as clean, truly quiet `ok`
					  snapshots promoting like `background`, and the built native packed
					  extension fixture. The verifier now rejects WorldFoam artifacts
					  with missing acceptance metadata, STAR compare rejects that
					  condition before spending a matched STAR run, and the wrapper
					  refuses to select a WorldFoam artifact with missing acceptance.
					  Train/eval and
					  STAR-compare benchmark capture now ignore the current process
				  ancestor chain, fixing self-contended `keyword:metal` preflights
				  when the launch wrapper mentions a `powerfoam_metal` config path.
			  Additional no-timing regression slices
		  also pass after the framebitmask hardening: the factorized selector plus
		  native packed/cutwalk compiler tests pass `24` tests, and the mixed
		  fused-slab MPS shader regression suite passes `8` tests. Rebuilt
		  real-ray smokes pass for direct shared replay, CSR affine moving rays,
		  slab affine VJP without ownerupdate, and slab per-track ownerupdate/VJP.
		  A failed slab ownerupdate artifact was traced to an unsupported smoke
		  invocation (`--include-ownerupdate` with default `--layout tiled`), and
		  the smoke now rejects that combination before writing a misleading
		  failed result. A fresh
			  clean-evening strict wrapper attempt reached one true-32f WorldFoam
			  diagnostic row before the external blocker restarted:
			  `2026-05-20_real32_strict_mini_wrapper_clean_evening.attempt1.worldfoam.json`
			  records `loaded_frame_count=32`, `repeat_loaded_frames=false`,
			  `3.104ms` total, `2.773ms` backward, train PSNR `12.987`, and heldout
			  PSNR `14.229`. The promotion summary
			  `2026-05-20_real32_strict_mini_wrapper_clean_evening.promotion_summary.json`
			  is still `worldfoam_preflight_failed_or_contended`: attempt 1 failed
				  post-run promotion after new live `ai_trader` offline TOTO MPS-export
				  monitors and transient `MTLCompilerService` appeared, attempt 2 never
				  launched WorldFoam because preflight stayed contended, and no STAR
					  compare command ran. The wrapper now separates the planned STAR
					  artifact from the selected STAR artifact: `star_compare_artifact`
					  stays null until a promotable STAR artifact exists, while failed
					  STAR attempts only populate latest-attempt/latest-written fields.
					  The verifier now requires exactly one promotable STAR attempt and
					  matching selected/latest STAR artifact paths.

Key notes/results:

- `agent_notes/loose_notes/2026-05-18_17-01-36_star_uvt_worldfoam_64px_scale_gate.md`
- `agent_notes/loose_notes/2026-05-18_17-17-12_worldfoam_highcap_24site_gate.md`
- `agent_notes/loose_notes/2026-05-18_17-34-42_worldfoam_highcap_insertfix_gate.md`
- `agent_notes/loose_notes/2026-05-18_17-46-33_worldfoam_localtape_and_nextfork.md`
- `agent_notes/loose_notes/2026-05-18_17-58-50_worldfoam_ownerrun_reverse_tape.md`
- `agent_notes/loose_notes/2026-05-18_18-13-45_worldfoam_ownerupdate_negative.md`
- `agent_notes/loose_notes/2026-05-19_16-30-57_owner_run_lean_recompute_positive.md`
- `agent_notes/loose_notes/2026-05-19_17-18-20_owner_run_packed_delta_storage_probe.md`
- `agent_notes/loose_notes/2026-05-19_17-11-27_owner_run_delta_packed_parity.md`
- `agent_notes/loose_notes/2026-05-19_17-20-46_owner_run_coeff_factorization_probe.md`
- `agent_notes/loose_notes/2026-05-19_17-37-33_owner_run_factorized_shader_fork.md`
- `research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_scale_64px_896t_vs_12site_2_4_8_16.json`
- `research_experiments/world_foam_lane2/results/2026-05-18_star_uvt_directatomic_vs_worldfoam_gate4_fusedmse_ownerrun_scale_64px_896t_vs_24site_2_4_8_16.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_owner_run_fused_mse_nomid_lean_scale_2_4_8_16_render16_site24_step3_warm1_clean_repeat_attempt3.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_owner_run_boundary_packed_delta_probe_render16_site24_2_4_8_16_v2.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_owner_run_coeff_factorization_probe_render16_site24_2_4_8_16.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_owner_run_delta_packed_factorized_recompute_nomid_ladder_2_4_8_16_render16_site8_contended.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_owner_run_delta_packed_factorized_recompute_nomid_ladder_2_4_8_16_render16_site8_clean_compare.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_frameselect_compare_blocked_live.summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_frameselect_compare_clean_site8_attempt1.factorized_frameselect_compare_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_frameselect_compare_clean_site8_attempt1.regular_factorized.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_frameselect_compare_clean_site8_attempt1.frameselect_factorized.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_blocked_live.factorized_frameselect_compare_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8.factorized_frameselect_compare_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8.attempt1.regular_factorized.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8.attempt2.regular_factorized.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry_interruptfix.factorized_frameselect_compare_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry_interruptfix.attempt1.regular_factorized.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry2_nonzero_retryfix.factorized_frameselect_compare_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry2_nonzero_retryfix.attempt1.regular_factorized.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry3_stalefieldfix.factorized_frameselect_compare_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry3_stalefieldfix.attempt2.regular_factorized.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry3_stalefieldfix.attempt2.frameselect_factorized.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry9_framebitmask_masksetup_quiet.factorized_frameselect_compare_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry11_framebitmask_parallelsetup.factorized_frameselect_compare_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site8_retry11_framebitmask_parallelsetup.attempt1.framebitmask_factorized.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site24_retry1_framebitmask_parallelsetup.factorized_frameselect_compare_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site24_retry1_framebitmask_parallelsetup.attempt1.regular_factorized.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_factorized_selector_compare_clean_site24_retry1_framebitmask_parallelsetup.attempt2.framebitmask_factorized.json`
- `research_experiments/world_foam_lane2/results/2026-05-19_worldfoam_framebitmask_render64_site24_i32offsets_2_4_steps8_warm4.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_framebitmask_render64_site24_i32_offsets_8f_contended_correctness.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_framebitmask_render64_site24_i32_offsets_16f_contended_correctness.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_framebitmask_render64_site24_selected_only_prep_4f_smoke.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_framebitmask_render64_site24_selected_only_native_cutwalk_4f_smoke.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_framebitmask_render64_site24_selected_only_native_cutwalk_2_4_8_16_path.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_preflight_font_blocked.promotion_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_preflight_verifier_integration_blocked.promotion_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_stable_preflight_blocked.promotion_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_starretry.promotion_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_starretry.attempt1.worldfoam.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_starretry.star_attempt1.star_compare.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_repeat32_framebitmask32fix.promotion_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_repeat32_framebitmask32fix.attempt2.worldfoam.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_repeat32_framebitmask32fix.star_attempt2.star_compare.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate.promotion_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate.attempt2.worldfoam.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate.star_attempt1.star_compare.json`
- `src/dataset_configs/multicam_val_deepview_03dog_128_8fps_32f.jsonc`
- `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc`
- `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_real32_native_cutwalk_loader_smoke.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_real32_dryrun.promotion_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_real32_strict_mini_wrapper_settle_retry.promotion_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_real32_strict_mini_wrapper_settle_retry.attempt1.worldfoam.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_real32_strict_mini_wrapper_settle_retry.attempt2.worldfoam.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_real32_preflight_toto_mps_blocker_check.promotion_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_real32_strict_mini_wrapper_clean_evening.promotion_summary.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_real32_strict_mini_wrapper_clean_evening.attempt1.worldfoam.json`
- `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_star_quality_bridge_native_cutwalk_microgate.json`
- `agent_notes/loose_notes/2026-05-19_20-11-12_owner_run_framebitmask_shader_fork.md`
- `agent_notes/loose_notes/2026-05-20_00-04-31_worldfoam_framebitmask_i32_offsets_and_prep_bottleneck.md`
- `agent_notes/loose_notes/2026-05-20_00-14-56_worldfoam_selected_only_prep_gate.md`
- `agent_notes/loose_notes/2026-05-20_00-40-03_worldfoam_native_ownerrun_cutwalk_fix.md`
- `agent_notes/loose_notes/2026-05-20_04-34-08_worldfoam_render96_site48_i32_gate_and_ai_export.md`
- `agent_notes/loose_notes/2026-05-20_04-41-35_worldfoam_real_frame_gate_contract.md`
- `agent_notes/loose_notes/2026-05-20_04-50-55_worldfoam_real32_fixture_smoke.md`
- `/tmp/worldfoam_factorized_i16_projection_smoke.json`
- `/tmp/worldfoam_factorized_i16_actual_smoke.json`
- `/tmp/worldfoam_factorized_frameselect_smoke.json`
- `/tmp/worldfoam_factorized_framebitmask_smoke.json`

Next useful experiment:

- For paper-math progress, build the cell-path fixture first: constant-density
  owner-run word, monoid scan, replay equivalence against per-frame WorldFoam,
  and finite differences for `DeltaTau`, `sigma`, color, and run length. Do
  not jump to boundary flux, flux witness scores, or Magnus compression before
  this passes. Implement the fixture from
  `research_notes/worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md`.
- For visual comparison progress, use
  `research_experiments/world_foam_lane2/run_visual_compare_three_lanes.py`.
  The tiny 64px/16f gate has visual artifacts for all three lanes, including
  direct disk media for `tokengs`, and now has one clean all-lane summary. The
  128px/16f medium tier and capacity tier are also green. Next do
  representation-specific quality rows rather than another blind scale-up:
  STAR can scale tubes/steps, WorldFoam needs a quality bridge beyond
  color-only fixed geometry, and dynamic 3DGS needs better
  initialization/camera/loss scheduling.
- Do not spend the next turn on another local replay micro-variant. The current
  frame-bitmask shader path is correctness-green through render64/site24/16f
  and now supports the signed-int32 bit-31 case needed for an explicit
  repeated-fixture 32f smoke,
  the selector gates already promote it over regular factorized at site8 and
  site24, exact native owner-run delta prep removes the dominant Python
  endpoint-sequence construction bottleneck, and the clean wrapper-plus-verifier
  STAR comparison now passes. The real 2/4/8/16f, synthetic repeated 32f, and
  render96/site48 i32-base-offset speed rows are recorded in `BASELINES.md`.
  The checked real32 DeepView fixture and one-step loader/shader smoke remove
  the immediate data-loader blocker for 32f, but not the clean timing or STAR
	  comparison gate. The warm settle retry and clean-evening retry narrow the
	  blocker further: warm WorldFoam 32f shader steps are around `2-3ms`, but the
	  live/offline `ai_trader` TOTO export can restart during or after an attempt
	  and contaminate post-run snapshots, preventing promotion and blocking STAR.
	  The quality bridge is now recorded: the native-cutwalk WorldFoam micro-gate
	  is speed-competitive against matched STAR, but it is not yet RGB-quality
	  competitive with STAR UVT or the solid same-source baseline (`12.248` best
	  train PSNR, `12.857` heldout PSNR, `17.575dB` train gap to STAR source).
	  The bridge now includes the existing render96/site48 capacity candidate and
	  records `capacity_candidates_improve_train_psnr=false`; that larger artifact
	  reaches only `9.875` best train PSNR and is worse on all overlapping primary
	  frames (`-2.55/-2.53/-2.27dB` train PSNR at `2/4/8f`), so naive render/site
	  capacity did not close the gap. The report also flags that this candidate is
	  missing the primary `16f` row instead of treating it as full-frame-set
	  coverage.
	  It also records whether the best-quality artifact is the matched-speed
	  artifact; a future quality-closing candidate now yields
	  `best_worldfoam_quality_needs_matched_speed_gate=true` until it gets its own
	  clean STAR comparison.
	  Default-preserving site initialization forks are now wired through Gate4 and
	  train/eval as
	  `--site-initialization {legacy_sparse,legacy_pixel_mean,legacy_frame_pixel_mean,legacy_frame_patch3_mean,stratified_grid,stratified_pixel_mean}`.
	  The legacy mode remains the default for artifact compatibility. Current
	  evidence is CPU-only but useful: direct initializer tests pass, CLI help
	  exposes the modes, and CPU Gate4 compiler smokes write `status=ok` for
	  `2/4f`. The CPU Gate1 quality reference at render16/site9/2f rejects naive
	  grid spread but promotes a better frame-local color fork for the next clean
	  run: legacy train/heldout PSNR is `11.862/12.671`, stratified is
	  `10.419/9.692`, `legacy_pixel_mean` is `13.025/14.614`, and
	  `legacy_frame_pixel_mean` is the best heldout CPU candidate at
	  `13.029/14.617`. A same-frame 3x3 patch fork,
	  `legacy_frame_patch3_mean`, is positive versus legacy sparse
	  (`12.761/14.315`) but worse than the one-pixel frame-local candidate. A
	  follow-up `stratified_pixel_mean` fork combines grid support with
	  train-sample mean color; it raises train PSNR to `13.679` but drops heldout
	  to `12.611`, so the updated bridge rejects it as train overfit. This means
	  image-cell coverage alone hurts, while averaging color at legacy support
	  points is positive; frame-local one-pixel averaging is still best among the
	  CPU initializer forks. There is no clean MPS PSNR/speed
	  artifact for `legacy_frame_pixel_mean` yet because the latest strict
	  preflight still returned `preflight_contended` with high-CPU `font_maker`,
	  high-CPU `ai_trader` monitor/check/export children, the periodic
	  `ai_trader` MPS exporter chain, and a torch queue wrapper active.
	  The generated site-initialization quality bridge report
	  `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_site_initialization_quality_bridge.json`
	  selects `legacy_frame_pixel_mean` as `next_mps_candidate`, records
	  `positive_candidate_count=3`, and rejects `stratified_grid` plus
	  `stratified_pixel_mean` as CPU-negative or heldout-negative forks.
	  The Gate4 affine candidate-CSR capacity probe now threads the same
	  `--site-initialization` option, and the tiny CPU artifact
	  `research_experiments/world_foam_lane2/results/2026-05-20_gate4_affine_candidate_csr_capacity_legacy_frame_pixel_mean_render8_site4_2_4f.json`
	  passes for `legacy_frame_pixel_mean` with candidate count scale `0.993x` and
	  storage scale `0.998x` over `2f -> 4f`.
	  The combined readiness report
	  `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_candidate_readiness.json`
	  gates the two artifacts together and currently writes
	  `ready_for_quiet_mps_quality_speed_run=true`, `quality_claim=false`, and
	  `speed_claim=false`. The fail-closed launcher
	  `research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py`
	  writes the planned real32 `legacy_frame_pixel_mean` command to
	  `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_frame_pixel_mean_plan.json`
	  and refuses to run train/eval when its strict preflight is contended; the
	  current executed preflight summary
	  `research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_verified.json`
	  is `preflight_contended`. The launcher now also promotes the blocker
	  summary to top-level fields (`preflight_benchmark_environment_status`,
	  blocker counts, reasons, and a compact process list) so future agents do
	  not need to dig through the full preflight stdout tail. It now also
	  supports `--preflight-stability-samples`; the regenerated plan requests
	  three clean preflight samples spaced by `5s` before a real train/eval run.
	  The refreshed executed preflight artifact records
	  `preflight_stability_samples_requested=3`,
	  `preflight_stability_samples_completed=1`, `preflight_stability_ok=false`,
	  `preflight_blocking_reasons=["high_cpu","keyword:torch","periodic_mps_exporter"]`,
	  and `preflight_blocking_process_count=8`; no train/eval artifact launched.
	  An earlier live stability sample also caught an active TOTO prediction
	  export as `keyword:mps`, so the monitor is still a timing-window blocker
	  even when its parent process is idle. The plan/preflight summaries now
	  include `result_verifier_command` for
	  `research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py`.
	  The verifier rejects the current preflight artifact because it is not
	  `train_eval_ok`, completed only `1/3` stability samples, has a contended
	  preflight, and has no MPS train/eval artifact. Use that verifier on the
	  first clean `legacy_frame_pixel_mean` run before claiming PSNR, speed, or
	  sublinear frame scaling. The launcher now also supports `--verify-result`;
	  the refreshed plan summary has `verify_result=true`, so future clean runs
	  can fail closed inside the launcher if the post-run verifier rejects the
	  artifact. It also supports retrying the whole stability sequence with
	  `--preflight-retry-timeout-s` and `--preflight-retry-poll-s`; unit coverage
	  proves the clean-after-dirty path, but the live retry smoke
	  `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_retrywait_smoke.json`
	  still stopped as `preflight_contended` before train/eval. Latest focused
	  verification is `py_compile` OK and `33` focused tests passing. A longer
	  strict retry execution,
	  `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_final_try.json`,
	  made `11` attempts over a `180s` retry window but still failed closed as
	  `preflight_contended`, completing only `1/3` stability samples and producing
	  no train/eval artifact. Verifier runs on `verified_retry2`,
	  `retrywait_smoke`, and `final_try` all fail with the expected
	  contended-preflight/missing-artifact failures. A later preflight-only
	  recheck,
	  `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_screen_blocker_recheck.json`,
	  also failed closed at sample `1/3` with high-CPU `font_maker`, high-CPU
	  `ai_trader` pytest/export children, the TOTO periodic exporter screen, and
	  a `keyword:torch` queue wrapper. The latest preflight-only artifact,
	  `research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_actionable_blockers.json`,
	  records the same blocked state with
	  `preflight_external_blocker_summary`: `2` high-CPU external jobs, `1`
	  torch worker, and `5` periodic exporter processes; the post-run verifier
	  rejects it as expected because no clean train/eval artifact exists. A fresh
	  current preflight artifact,
	  `research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_current_preflight.json`,
	  also failed closed before training with high-CPU `font_maker`, ai_trader
	  pytest/export children, the torch queue wrapper, and the TOTO exporter
	  chain. A later status recheck
	  `research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_next_mps_current_status_recheck.json`
	  still fails closed, now with high-CPU `font_maker` PID `92641` at
	  `209.2%` CPU, the `keyword:torch` queue wrapper, and the same TOTO exporter
	  chain. The
	  source-only native variant verifier
	  `research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_native_variant_source_wiring.json`
	  passes for the `fused_direct`, `fused_csr`, and `fused_slab` forks. It
	  checks `TORCH_LIBRARY` schemas, `m.impl` registrations and dispatch-target
	  source definitions, Python `torch.ops` wrappers, host-loaded Metal kernel
	  names against kernels declared in the dynamically loaded `.metal` source
	  files, and `MetalKernels` field declarations/initializers/uses. The three
	  Python wrappers now load their pure `TORCH_LIBRARY` binaries with
	  `torch.ops.load_library`; the import verifier
	  `research_experiments/world_foam_lane2/results/2026-05-21_worldfoam_native_variant_import_registration.json`
	  proves normal package import registers all compiled schemas: direct
	  `11/11`, CSR `13/13`, slab `103/103`; after rebuilding all three forks
	  from source, that artifact records fresh rebuilt shared-library mtimes.
	  Rebuilt MPS correctness smokes also pass for direct/CSR/slab
	  power-boundary counts:
	  `2026-05-21_worldfoam_rebuilt_direct_power_boundary_mps_smoke.json`,
	  `2026-05-21_worldfoam_rebuilt_csr_power_boundary_mps_smoke.json`, and
	  `2026-05-21_worldfoam_rebuilt_slab_power_boundary_mps_smoke.json`. The
	  slab mixed MPS regression suite passes `8` tests over ownerupdate,
	  sample-reduce, framegroup cached, and high-cap replay kernels. The invalid
	  `--include-ownerupdate` plus default-`tiled` smoke invocation now has a
	  focused parser regression. The rebuilt-native smoke-bundle verifier
	  `2026-05-21_worldfoam_rebuilt_native_smoke_bundle_verifier.json` passes,
	  requiring seven valid rebuilt smoke artifacts and classifying the old
	  failed tiled-ownerupdate artifact as `expected_invalid_tiled_ownerupdate`.
	  The goal-state report
	  `2026-05-21_worldfoam_fork_shader_goal_state.json` records
	  `shader_fork_smoke_state_fixed=true` but `objective_complete=false` and
	  `status=blocked_external_environment` because the clean real32 MPS
	  PSNR/speed/sublinear gate still has no artifact. Commit/handoff scope is
	  recorded in
	  `research_experiments/world_foam_lane2/2026-05-21_worldfoam_fork_shader_commit_scope.md`,
	  including the generated native outputs to exclude from a source commit.
	  Current source/import/rebuilt focused verification is `51` tests passing
	  plus the `8`-test MPS slab suite; this is wiring/import/kernel-smoke
	  evidence, not runtime PSNR or speed evidence. A
	  later fresh preflight-only artifact,
	  `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_fresh_preflight.json`,
	  also failed closed at sample `1/3`; it recorded `8` blocking rows,
	  including high-CPU `font_maker`, high-CPU `ai_trader` pytest/report
	  children, the TOTO monitor chain, and a `keyword:torch` queue wrapper.
	  Follow-up probe
	  `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_probe2351.json`
	  also failed closed at sample `1/3`, now catching a live TOTO quote
	  snapshot and multiple high-CPU pytest/RL children. Probe
	  `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_legacy_pixel_mean_probe2354.json`
	  also failed closed at sample `1/3` with high-CPU `font_maker`, high-CPU
	  `ai_trader` imitation/integrity pytest children, the TOTO monitor chain,
	  and a `keyword:torch` queue wrapper.
	  The next useful work is therefore either a quiet full wrapper run using the
	  real32 config plus matched STAR video after pausing/stopping those monitors,
	  or a WorldFoam quality/capacity run that tries to close the RGB gap while
	  preserving the clean speed gate. Local timing gates should either pause/stop
	  the ai_trader TOTO monitor or keep using strict retry wrappers, because its
	  per-iteration export can briefly consume CPU/Metal and contaminate an
	  otherwise clean run. Use
  `--require-real-loaded-frames` on any future long-frame promotion run so a
  synthetic repeated-fixture smoke cannot be mistaken for real temporal scale.

- To reproduce or refresh the accepted micro-gate, use the wrapper plus
  verifier:

  ```bash
  rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
    research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py \
    --run-id <new-run-id> \
    --wait-timeout-s 300 \
    --wait-poll-s 30 \
    --max-worldfoam-attempts 3 \
    --max-star-attempts 3 \
    --preflight-stability-samples 3 \
    --preflight-stability-interval-s 5 \
    --verify-promotion
  ```

## PowerFoam Post-Audit

Status: **Parked unless user selects the lane**

Purpose: continue the post-audit PowerFoam research backlog after local Metal
trainability/parity gates.

Key docs:

- `TODO/powerfoam_remaining_work_after_completion_audit_2026-05-06.md`
- `TODO/powerfoam_full_reproduction_todo.md`
- `BASELINES.md` PowerFoam rows

Current decision:

- Do not mix this lane with V-JEPA/token-GS work by default.
- Avoid another shallow sweep unless it targets the structural/depth/material
  blockers named in the TODO and key learnings.

## Browser WebGPU Adam And Density Control (2026-07-10)

- Surface: `web/dynaworld_browser_trainer/`, asset tag `converge59`.
- Implemented: per-parameter Adam moments; absolute center-gradient,
  contribution, opacity-gradient, and motion-gradient EMAs; fixed-cap recycling
  every 256 steps; localized residual births; optimizer/recycle diagnostics;
  serialized readback; validation-time train pause; startup reset guard.
- Rejected calibration: the first Adam mapping reused SGD-scale rates. At 320
  splats the grid loss worsened `0.000369 -> 0.000400`; this was rejected and
  all parameter-group rates were reduced by 10x.
- Diagnostic: a 96-splat probe confirmed the optimizer was active (`Param
  Delta 5.88e-2` by step 526 under the too-large mapping) rather than silently
  frozen. Later low-rate probes were finite and nearly neutral, not a quality
  win. Broad births were also rejected; final births use radius `0.010`, eight
  slots/event, and high-residual motion samples.
- Verification: JS syntax and diff checks pass; localhost returns `200`; the
  in-app browser compiled the WGSL pipeline on the Apple adapter and reached
  `Ready` with no `converge59` warnings/errors. Do not record a `BASELINES.md`
  row: no matched quality promotion was established.
- Next falsification gate: tiled/image-space backward plus true windowed
  D-SSIM, then a matched source-view ablation against `converge47`; novel-view
  claims still require a calibrated multicam objective.

## Browser Calibrated Multicamera Demo (2026-07-19)

- Surface: `web/dynaworld_browser_trainer/`, asset tag `multicam67`.
- Scope: demo/prototype only. It does not add a Python trainer, modify the
  unified paper runner, or claim native World Tubes/dynamic-3DGS parity.
- Data: the existing `src/train/export_dynaworld_browser_bundle.py` now has a
  thin dataset-export mode that calls `load_multicam_video_bundle` for the
  canonical full-300-frame Coffee Martini manifest. It exports eight exact
  times `[0,43,85,128,171,214,256,299]` at `96x72`, with `cam04`/`cam09`
  train and `cam06` heldout validation-only. Three small PNG atlases replace
  browser MP4 seeking, which had silently returned the same frame at every
  requested time. Initialization uses 768 visible anchor-frame XYZRGB points
  from the existing Ex4DGS SfM `input.ply`; no heldout pixels seed parameters.
- WebGPU: shared 3D primitives are projected through canonical normalized
  intrinsics and anchor-relative poses. Both simplified motion modes run in the
  isolated `trainerWebGpu3d.js`. The compute shader was reduced to Apple's
  eight-storage-buffer stage limit by packing sample indices and removing
  redundant GPU metric/background buffers.
- Browser evidence on the Apple adapter: the app loads with no app errors,
  shows all three synchronized camera views, switches the main target/render to
  `cam06 (heldout)`, and produces distinct frame-0/frame-7 target pixels.
  Motion sampling finds `5,635` train spacetime pixels. Revised SfM radius and
  opacity initialization starts at train/heldout loss
  `0.182629/0.192498`, PSNR `7.4/7.2 dB`, and coverage `32.6/32.9%`.
  A 132-step World Tubes-style trace reaches train/heldout loss
  `0.173302/0.185356`; a separate 119-step dynamic-splats-style trace under the
  earlier sparse init also decreased both losses. Global-luma SSIM remains a
  validation proxy, not a windowed SSIM implementation or training loss.
- Verification: `11` focused paper-protocol/browser-adapter tests pass. No
  `BASELINES.md` row is warranted; quality remains low and compositing is fixed
  order without depth sorting. The next admissible quality work is tiled
  image-space backward, depth-aware composition, and true windowed SSIM under a
  matched train/heldout ablation.

## Adding A New Experiment

For every experiment worth resuming later:

1. Add or reuse a checked-in JSONC config under `src/train_configs/`.
2. Put launch commands in `src/train_scripts/` when they are reusable.
3. Write logs to `outputs/run_logs/` or a lane-specific `results/` folder.
4. Record W&B ids, result JSONs, and media paths in this file.
5. Add a `BASELINES.md` row only when it is a benchmark/baseline claim.
6. Add a dated loose note for chronology and failures.
