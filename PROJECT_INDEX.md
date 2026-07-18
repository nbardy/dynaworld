# DynaWorld Project Index

This is the first operational index after `AGENTS.md`. It is for a new agent
that has no thread context and needs to know where the project state, active
experiments, logs, TODOs, and code-organization rules live.

## Five-Minute Startup Path

Read these in order:

1. `AGENTS.md` - repo rules, smoke gates, config style, notes policy.
2. `PROJECT_INDEX.md` - this file; the map of maps.
3. `.agents/thread_types/README.md` - evergreen thread types, including the
   hourly CTO code reviewer.
4. `TODO/README.md` - current backlog and active next steps.
5. `EXPERIMENTS.md` - active experiment lanes, configs, logs, W&B/result files.
6. `BASELINES.md` - canonical benchmark rows and reruns needed.
7. `research_notes/data_contract.md` - data-loader contract for same-view and
   novel-view training.
8. `CODE_ORGANIZATION.md` - modularity and deduplication roadmap.
9. `agent_notes/key_learnings.md` - compressed technical lessons.
10. Latest relevant `agent_notes/loose_notes/*.md` - raw chronology and handoff.

If the task is architecture or new math, also read `research_notes/README.md`
and the routed strategic docs it names before proposing a new direction.

## Key Index Files

| File | Purpose | Update when |
| --- | --- | --- |
| `AGENTS.md` | Agent operating rules and required gates. | Rules, gates, or startup routing change. |
| `.agents/thread_types/` | Evergreen agent roles and schedulable review/reporting workflows. | A reusable thread type or cadence is added or changed. |
| `PROJECT_INDEX.md` | High-level map for new agents. | A new canonical index/doc surface appears. |
| `TODO/README.md` | Active backlog and next-step routing. | A lane closes, supersedes, or changes priority. |
| `EXPERIMENTS.md` | Active experiment registry with configs/logs/results. | Any key experiment starts, stops, or gets promoted. |
| `BASELINES.md` | Canonical measured baseline/benchmark table. | A benchmark row is run or re-run. |
| `CODE_ORGANIZATION.md` | Refactor and modularity roadmap. | Code organization priorities change. |
| `research_notes/data_contract.md` | Same-view vs multicam/heldout data contract. | Data loaders, manifests, or eval semantics change. |
| `agent_notes/key_learnings.md` | Dense memory bank of surprising lessons. | A failure changes how future agents should reason. |
| `agent_notes/loose_notes/` | Raw dated session notes and closeouts. | Every meaningful session/work chunk. |

## Current Project State

As of 2026-07-19, the camera-gauge and ray-fiber mathematics is retained, not
closed. The stopped lane is only open-ended theory/name proliferation without
a measured compiler failure. Its primary method and paper surface is **World
Tubes**, implemented by projective **STAR UVT**: the camera-ray bundle
invariant, event-certified gauge domains, compiled interval atlas, visibility
strata, and direct adjoint path remain mainline. **WorldFoam** is a real
second-paper retained-depth operator ordering, but is parked as an engineering
priority until broader heldout quality or native optical-transfer parity
justifies another systems push. The canonical naming and lane map is
`research_notes/renderer_lane_taxonomy.md`. The original closeout evidence and
repository-accounting decision are recorded in
`agent_notes/loose_notes/2026-07-17_23-59-22_world_tubes_lane_closeout_and_repo_integration.md`.

As of 2026-07-11, the first matched real-dataset paper table is complete on
Neural3D `coffee_martini`: train `cam04`/`cam09`, hold out `cam06`, 128px/16f,
40 steps, 1024 primitives, and seeds 17/29/43. The verified report is
`outputs/benchmarks/2026-07-11_coffee_martini_matched_sweep/report/summary.json`.
World Tubes leads mean heldout PSNR on this single split. The next benchmark
step is camera-triplet and scene breadth; do not generalize this one-split row
into a multi-scene SOTA claim. The top-level combined evidence report is
`outputs/benchmarks/2026-07-11_paper_runner_table_report/summary.json`; it now
requires this matched multicamera table in addition to the earlier fixtures
and local-video capacity smoke.

As of 2026-05-28:

- Gauged/projective STAR UVT now has a machine-checked compiled-adjoint
  replacement artifact for the practical real-video trainer route. The report
  proves the trainer selects the projective interval route, the harness lowers
  forward to interval Metal and backward to the interval Metal direct VJP with
  visibility/tile membership held as compiled constants, and the broad10
  real-video case payloads preserve renderer gradients with clean cache/support
  behavior. The final completion audit at
  `outputs/benchmarks/2026-05-25_star_uvt_projective_goal_final_completion_audit/summary.json`
  consumes the progress and gap reports, verifies the ten theory subfolders,
  proves nine objective-level rows, records
  `final_goal_completion_accepted=true`, and promotes the previous
  `full_goal_completion` gap into an accepted completion claim.
- STAR UVT feature-tube support has a repaired sparse binner for chunk-shifted
  moving tubes. The selected-patch diagnostic found valid analytic support was
  being culled; `tube_bounds` now uses inverse-precision bounds with a smaller
  determinant tolerance and fallback bounds that cover shifted local chunks.
  The first targetarea2 binfix 50-step row passes fixed-bin with zero overflow,
  max tile `110/128`, loss `0.889263 -> 0.863064`, support-target-area loss
  `0.253626 -> 0.217254`, and selected-patch PSNR
  `6.644/19.452/26.994` normal/forced/oracle. Dense support is now measured at
  `7.269/14.736/21.439` normal/forced/oracle PSNR with alpha `>0.1` `75.4%`.
  Prefix tape shows selected born support is present and not meaningfully
  hidden on selected target rays: selected weight share `93.1%`, top selected
  contributor `95.7%`, prefix-hidden `1.6%`. The prefix-alpha follow-up passes
  fixed-bin and moves selected-ray contribution (`0.4114 -> 0.4419`) plus final
  alpha (`0.4456 -> 0.4751`) over 50 steps, but dense support is effectively
  unchanged at `7.262/14.732/21.438` normal/forced/oracle PSNR with alpha
  `>0.1` `75.4%`. The fix is real but still coverage/alpha/composition
  limited; the next STAR gate should broaden support ownership/coverage or
  change the sampling distribution, not repeat local alpha pressure alone.
- STAR UVT has a first-class `src/train/train.py` route for source-view
  overfit. `direct_atomic + index_add` is the practical 64f path. Deterministic
  compact backward is still blocked on load-growth/backward speed.
- 2026-05-20 STAR UVT feature-tube routing: the shader diagnostic phase has a
  fast practical route, but visual quality blocks scale-up. Compact target-area
  reaches only `6.023` dense full RGB PSNR, RGB-grid-only reaches `5.657`, and
  compact+RGB-grid40 reaches `5.720`, all with sparse/streaked media versus the
  RGB STAR same-clip bracket at `12.444`. The dense alpha diagnostic localizes
  this as a coverage/visibility/composition failure: forced alpha reaches
  `11.450-14.616` PSNR and target-background oracle composition reaches
  `20.149-25.562`, while alpha `>0.1` covers only `41.5-43.5%` of pixels. The
  direct alpha-to-one coverage gate is also rejected: it improves sampled alpha
  loss `0.752440 -> 0.738210`, but dense RGB stays `6.018`, alpha `>0.1` stays
  `43.1%`, and feature/probe losses regress. The phase-covered alpha retry is
  rejected too: dense RGB falls to `6.014`, dense alpha `>0.1` falls to
  `43.0%`, and feature/probe losses still regress. A target-aware black-hole
  coverage retry is also rejected: it improves its own loss `0.262537 ->
  0.256889`, but dense RGB stays `6.014`, alpha `>0.1` stays `43.0%`, and
  feature/probe losses regress. Target-background composition is informative
  but also rejected: it raises forced-alpha PSNR to `14.891-14.899` and oracle
  composition to `27.105-27.443`, but black-background dense RGB is only
  `5.666-5.748` and alpha `>0.1` stays `40.8-43.1%`. The alpha-sweep/patch4
  follow-up is also rejected: `16x` posthoc alpha gain reaches only
  `8.337-8.592` PSNR, and `4x4` support plus target-background alpha pressure
  ends at `5.698` dense RGB while regressing feature/probe. The raw-opacity
  bias render sweep is negative too: best logit bias `+4` only reaches
  `6.194/5.926/5.871` PSNR for compact/targetbg-alpha/patch4 and barely moves
  alpha `>0.1` coverage. Dense alpha-only support is also rejected: the new
  `dense_alpha` trainer path adds `834.5/124.6/858.9ms` render/loss/backward,
  fails loss decrease `1.271702 -> 1.284505`, drops probe PSNR
  `22.028 -> 21.861`, lands at `5.647` dense RGB, and lowers alpha `>0.1` to
  `40.7%` despite `14.556/25.809` forced-alpha/oracle PSNR. The follow-up
  alpha-only visibility profile is only a speed implementation detail: sparse
  all-pixel F1 alpha render plus cached F1 backward matches dense alpha exactly,
  matches gradients within `4.7e-7`, and cuts alpha render+backward
  `1100.8 -> 634.6ms`. Wired into the trainer as
  `dense_alpha.render_mode=sparse_f1`, it cuts mean step/backward
  `2558.6/1114.2 -> 873.3/370.0ms` and dense-alpha render/loss/backward
  `834.5/124.6/858.9 -> 276.0/22.0/303.7ms`, but it does not change the failed
  objective. Do not
  launch the 300-video STAR UVT scale lane until a stronger visibility-aware
  objective/model bridge clears dense media.
- The first support-changing visibility bridge now has both a CPU mechanism
  gate and a first-class trainer mechanics gate, not a quality promotion.
  `visibility_support_bridge_prototype.py` starts from a miss where target
  alpha `>0.10` is `0.0`; same-support dense alpha training stays at `0.0`,
  while the soft projected-tube coverage proxy reaches target alpha `>0.10`
  coverage `0.324`. The trainer port then passes from the sparse 1500
  checkpoint with weighted loss `0.871986 -> 0.871864`, RGB-probe PSNR
  `22.0277 -> 22.0291`, center/velocity gradients seen, and `237ms` mean proxy
  overhead. It still does not prove dense visual quality or scale readiness
  because dense full RGB PSNR is `5.640`. The support diagnostic rejects the
  current center-only proxy as the next scale bridge: forced-alpha/oracle
  content improves, but alpha `>0.1` moves `41.1% -> 40.5%`; a 10x/20-step
  proxy run fails trainer loss and only reaches `40.6%` alpha `>0.1`.
  The follow-up opacity/precision support-aware proxy is now wired and tested,
  but it is also not the bridge: it sends raw opacity/precision gradients and
  lowers support proxy loss `3.4303 -> 3.3821`, while feature loss slightly
  worsens, dense RGB only moves `5.640 -> 5.643`, alpha `>0.1` stays
  `40.5% -> 40.6%`, and proxy work costs `693.7ms`/step.
  The next mechanism gate is positive on CPU: fixed-budget birth/split
  reallocates `8/16` dead tubes onto target support, taking target alpha
  `>0.10` to `1.0000` while same-support alpha stays at `0.0000`; refinement
  keeps `1.0000` and lowers background alpha `0.0479 -> 0.0072`. The trainer
  port now exists as `support_birth_split.enabled` and passes a 64f/512px gate:
  `32/8192` low-opacity tubes are reallocated from the sparse 1500 checkpoint,
  selected opacity rises `0.3418 -> 0.8000`, zero overflow holds
  (`100/71/128` max/p95/cap), and 5-step timing is `189.4ms` mean /
  `138.3ms` last with full RGB PSNR `5.708`. This is a Metal trainer primitive,
  not a visual-quality claim. The dense-support diagnostic confirms that split:
  birth32 has the best normal/forced-alpha/high-alpha support among the current
  5-step support rows (`5.708` normal, `14.606` forced-alpha, alpha `>0.5`
  `0.117`), but alpha `>0.1` is only `0.411` and target-background oracle falls
  to `25.234`. The uncovered-brightness target sampler follow-up is now run:
  it selects genuinely low-alpha bright samples (`selected_alpha_mean=0.0209`),
  passes at `187.4ms` mean step, and reaches `5.713` dense RGB PSNR, but still
  leaves alpha `>0.1` at `0.411` with forced-alpha PSNR `14.579`. The next gate
  is now partially run: cap `128` overflows at `64+` births, cap `256` clears
  `64/128`, and radius `96px` raises coverage more than tube count/source.
  Best safe cap-128 row is `low_alpha_n32_r96_cap128` with alpha `>0.1`
  `0.420`, normal PSNR `5.825`, forced-alpha PSNR `14.591`, oracle `24.226`,
  and max tile `100/128`. This is still not a quality promotion because oracle
  falls. The intermediate-radius follow-up confirms that it is a smooth
  tradeoff, not a hidden sweet spot: uncovered `r64/r72/r80/r88` raises alpha
  `>0.1` `0.411 -> 0.413 -> 0.415 -> 0.417` while oracle falls
  `25.319 -> 25.187 -> 25.015 -> 24.802`, and low-alpha `r80/r88` fails the
  loss gate despite zero overflow. Next STAR support work should change
  born-tube initialization, not run longer on the current radius schedule.
  The scalar opacity init sweep is now also negative: `r80` uncovered opacity
  `0.4/0.6/0.8/0.9` moves alpha `>0.1` only `0.414 -> 0.415` while oracle
  falls `25.177 -> 24.987`; `r88` opacity `0.2/0.4/0.6/0.8` moves alpha
  `0.414 -> 0.417` while oracle falls `25.242 -> 24.802`. The first
  support-shape gate is now negative too: single-line `trajectory_ellipse`
  support passes eight cap-128 rows with zero overflow but only reaches alpha
  `>0.1` `0.408-0.409`, below the prior isotropic `0.411`. Next support work
  should be multi-center or stratified birth/split. The first multi-center gate
  is positive: `farthest_xy` with `K=8`, `32` births, `r64`, and cap `128`
  reaches alpha `>0.1` `0.4309` and alpha `>0.5` `0.1550` with zero overflow
  (`101/71/128`) and forced-alpha PSNR `14.608`, but oracle drops to `23.965`.
  The K8 radius/opacity sweep selects a better balance: `r64/o0.4` keeps alpha
  `>0.1` at `0.4298`, forced-alpha `14.620`, oracle `24.805`, zero overflow,
  and `167.9/58.1ms` step/backward. Next support work should run the 20-step
  media gate for that row. That gate now passes and holds the coverage gain:
  loss `0.903197 -> 0.897231`, probe PSNR `21.681 -> 21.769`, zero overflow,
  last step/backward `147.3/54.3ms`, dense alpha `>0.1` `0.431158`,
  forced-alpha `14.631`, oracle `24.851`. It is still not a visual-quality
  solution. The matched `K=8/r72/o0.4` 20-step comparison also passes with zero
  overflow, dense alpha `>0.1` `0.432454`, full RGB PSNR `5.820`, and
  `157.9/61.1ms` mean step/backward, but loses feature/probe loss and oracle
  (`24.851 -> 24.668`). The 50-step regenerated-checkpoint continuation then
  split the routing: `K=8/r64/o0.4` improves loss/probe but fails cap-128
  (`277` overflowed tiles, max `146/128`), `K=8/n16/r48` and `K=8/n16/r40`
  reduce that to only two overflow tiles (max `131/128`), and
  `K=8/n8/r40/o0.4` is the current cap-safe seed (`pass=true`, zero overflow,
  max `123/128`, loss `0.754568 -> 0.749460`, RGB-probe PSNR
  `24.372 -> 24.501`). The longer safe-row gate selects the 90-step checkpoint
  (`pass=true`, zero overflow, max `122/128`, loss `0.754568 -> 0.747006`,
  feature loss `0.608402 -> 0.606764`, RGB-probe PSNR `24.372 -> 24.552`);
  the 100-step sibling remains fixed-bin but fails after late objective jumps.
  The checkpoint-aware 100-step tail schedule passes (`0.749454` final loss,
  zero overflow, max `122/128`) but does not beat the selected 90-step
  checkpoint and matches its dense support profile. Dense support improves over
  `start1500`
  (`6.035 -> 6.472` normal PSNR, `10.702 -> 14.018` forced-alpha,
  `16.787 -> 21.602` oracle) but is nearly flat across 50/90/100, so the
  normal/forced/oracle gap keeps this as support progress rather than a
  visual-quality closeout. The uniform/all-centers follow-up is also measured:
  uniform `n16`, `K=16/n16`, and `K=16/n16/r32` still overflow by two tiles,
  while `K=12/n12/r40/o0.4` is cap-safe but does not beat the selected
  `K=8/n8` 90-step checkpoint or change dense forced-alpha/oracle support. The
  first cap-aware bridge is now measured: cap-slack target scoring alone still
  hits the same two-tile overflow, exact-fit repair drifts to one final overflow
  tile, and guarded repair (`K=16/n16/r40/o0.4`, guard `2`, four dropped born
  tubes) passes fixed-bin with max `127/128` and dense normal/forced/oracle PSNR
  `6.486/14.021/21.571`. This is a useful cap-safe mechanism but only a tiny
  support nudge over `K=12/n12`. The first residual-cap-slack scorer improves
  scalar loss/probe a little while preserving fixed-bin (`0.753586 -> 0.748839`,
  max `127/128`), but dense support stays flat (`6.486/14.019/21.579`
  normal/forced/oracle). The footprint-aware residual scorer is also measured:
  it is the best K16 scalar row (`0.752912 -> 0.748672`) but dense support stays
  flat (`6.481/14.021/21.576`). Target-grid feature init for born support is now
  a small positive (`0.752454 -> 0.748504`, dense `6.488/14.054/21.629`) but
  alpha coverage remains flat (`>0.1` `0.655`). The first support-target alpha
  bridge also passes and learns its local pointwise objective
  (`0.492962 -> 0.478448`), nudging dense support to
  `6.508/14.084/21.626` and alpha `>0.1` `0.657`, but the forced-alpha/oracle
  gap remains the same class of failure. The support-target-area 2x2 patch
  bridge is a cheaper local positive (`0.597970 -> 0.581641`, dense
  `6.507/14.085/21.627`, alpha `>0.1` `0.657`) but does not beat that plateau
  and weakens feature loss versus target-init. The next STAR UVT support gate
  needs visibility-prefix/compositing tape behavior, not dataset scale-up,
  schedule-only cleanup, tile repair, target scoring, pointwise alpha, or small
  target-area patches alone.
- STAR UVT feature tubes have a passing dense Gate 0 contract on CPU/MPS and a
  first direct feature Metal path with tiny F=4/F=32 parity. The usable
  64f/256px/32768/F32 row is `757.9ms` total / `567.8ms` backward with zero
  overflow. The first-class `arch=star_uvt_feature_overfit` trainer/config path
  now passes a real-video 8f/64px/512t/F32 mini-overfit smoke
  (`0.18602 -> 0.04167` loss in 20 steps). Frame-chunked autograd parity passes
  with sub-micro gradient errors, and the chunked first-class smoke matches the
  same loss curve with `frame_chunk_size=2`. First 64f scale probes pass with
  zero overflow (`256px/8192t/chunk4` at `0.965s/step`, `512px/2048t/chunk2` at
  `4.021s/step`), but both are backward-dominated and not quality baselines.
  Higher 256px tube-count diagnostics expose the current real-video overflow
  wall against the 128-entry tile cap: 16384 tubes overflows `736` tiles
  (max `151`, p95 `123`), while 32768 overflows `8160` (max `274`, p95 `238`).
  Cap 256 makes 16384 valid and makes 32768 valid only with support pruning;
  unpruned 32768 still overflows `216` tiles at cap 256. The current best valid
  32768-tube feature candidate is `alpha>=1/72/cap256`: 20 steps improve loss
  `0.31889 -> 0.29290`, PSNR `4.96 -> 5.33`, at `1.321s/step` with
  `1.021s` backward and zero overflow. It is only four refs under the cap
  (`max tile 252/256`), so `alpha>=1/64/cap256` remains the conservative
  zero-overflow fallback (`max 248`, loss `0.29350`). `alpha>=1/80` and
  `alpha>=1/96` have slightly better 20-step loss but overflow late. This is
  still a speed/validity row, not a quality replacement for RGB STAR.
  `feature_uvt.render_mode=feature_direct_fixedbin` now gives the trainer a
  requested/fallback contract, but it is explicitly an alias for the direct
  feature kernel until a native fixedbin backward exists. New outputs record
  `kernel_backward_mode=direct_atomic` and
  `requested_fixedbin_is_direct_atomic_alias=true` for that request. Unpruned
  `32768t/cap256` still falls back after `216` overflow tiles, while
  `alpha>=1/72/cap256` is zero-overflow eligibility evidence, not a distinct
  optimized fixedbin shader. `feature_direct_gradcache` is now the first actual
  feature fast-backward mode; it caches the pixel grad vector and gives a modest
  serial synthetic backward win (`485.63ms -> 471.29ms`) plus a first-class
  passing row at `1.226s/step` and `0.973s` backward. A benchmark-only
  skip-feature-gradient diagnostic cuts synthetic backward to `327.71ms`, so
  the next real shader should reduce feature-gradient atomics or move to an
  RGB-grad handoff. The first trainable reduced-gradient prototype
  (`feature_direct_gradcache_reduce`) passes parity and the 20-step row, but it
  is slower than plain gradcache (`1.261s/step`, `1.000s` backward versus
  `1.226s/step`, `0.973s`), so it remains a recorded negative result rather
  than the default. The vectorized follow-up
  (`feature_direct_gradcache_reduce_vec4`) also passes parity and improves one
  synthetic cap128 control, but the real cap256 first-class row is slower than
  gradcache and scalar reduce (`2.095s/step`, `1.413s` backward versus
  `1.807s`/`1.333s` and `1.890s`/`1.395s` on the rerun), so it remains a
  diagnostic mode. The cached-bin sidecar (`gradcache_cached_bins`) reuses
  forward tile bins and is a bounded synthetic win (`1068.0ms -> 935.8ms`
  renderer backward on the same-session 64f/256px/32768t/F32 pair), but the
  first-class 512px/8192t/chunk2 trainer row ties step time and has slower
  measured backward than plain gradcache (`16.20s/10.24s` versus
  `16.21s/9.68s`), so it is diagnostic only. The sequential direct-mode matrix
  runner now lives at
  `research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py`
  and now records cached-bin modes plus `kernel_backward_mode`/`cached_bins`.
  The 128/256/512 all-mode table lives under
  `outputs/benchmarks/2026-05-19_star_uvt_feature_direct_mode_matrix_128_256_512_64f_32768t_f32_cached/`;
  all 39 rows pass. At 512px, `gradcache_cached_bins` is the fastest
  full-gradient direct total row (`1.979s`, `1.103s` backward), but
  `gradcache_skip_feature_grad` is still the fastest diagnostic (`1.714s`,
  `0.804s` backward), so feature-gradient accumulation remains the shader target.
	  The older dense-analytic target-grid/frozen-probe trainer render-mode matrix lives at
	  `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix.md`
	  plus `..._repeat_top.md`; it confirms all current modes train to the same
	  5-step loss/probe PSNR under dense analytic VJP, but reduce/vec4 do not win
	  that path end-to-end and fixedbin-request is only the direct-atomic alias.
	  The sparse-grid matrix and sparse-forward trainer gate supersede it for the
	  selected speed path.
  A benchmark-only `gradcache_feature_grad_only` /
  `gradcache_two_pass_feature_grad` gate now passes tiny F4/F32 parity, but
  naive split-recompute is a negative speed result: refreshed 256px full
  gradcache is `0.972s` total / `0.692s` backward versus two-pass `1.343s` /
  `1.063s`; refreshed 512px is `2.467s` / `1.379s` versus two-pass `2.471s` /
  `1.613s`, and the reverse-order 512px check stayed negative. Treat
  "two-pass" as a true fixedbin/tile-slot accumulation design or native VJP
  handoff, not duplicated traversal.
  The follow-up tile-slot budget gate says the reason is precise: a tile-slot
  feature accumulator can cut feature-gradient write count by `128x`, but naive
  prefix recompute is `39.8x` slot-pixel work at 256px and `10.8x` at 512px,
  while a scalar contribution-weight tape is already `1.17-1.20GiB` and a
  per-channel tape would be `37-38GiB`. Any implementation should therefore
  materialize compact scalar weights/prefixes or move the VJP boundary, not
  store per-channel weights or recompute prefixes per slot.
  The reducer-only isolation gate then showed the current barrier-heavy
  tile-slot reducer has a usable core: `gradcache_feature_grad_only_reduce_vec4`
  cuts feature-only backward `532.8 -> 449.9ms` at 256px and
  `869.1 -> 774.8ms` at 512px. A full-gradient same-session refresh also has
  `gradcache_reduce_feature_grad_vec4` beating plain gradcache at 512px
  (`1284.2 -> 1108.0ms` backward), but two-pass compositions still lose or tie
  because duplicate traversal remains dominant.
  A narrow benchmark-only RGB handoff prototype
  (`fused_first3_sigmoid_mse`) is positive: it computes
  `alpha * sigmoid(feature[:3]) -> mean MSE` VJP inside Metal, passes F4/F32
  parity, and times at `309.31ms` synthetic backward versus `547.58ms`
  same-session gradcache. The generalized in-tile linear-colorizer handoff now
  exists and passes parity with weight/bias gradients, but it is slower than
  gradcache (`615-619ms` backward versus `477.5ms` same-session gradcache);
  even skipping colorizer grads was noisy (`598.5-714.1ms`). Do not promote it
  to the trainer as-is. The image-space-prep logit handoff also passes parity
  but is slower than gradcache (`595.2ms` renderer backward plus `60.2ms` prep
  versus `529.0ms` same-session gradcache). A follow-up combines that logit
  handoff with the existing stable-tile reducer modes:
  `logit_handoff_reduce_vec4` passes F4/F32 parity and zero-overflow 256/512
  direct rows, improving synthetic backward `571.7 -> 510.6ms` at 256px and
  `654.8 -> 642.3ms` at 512px. Scalar reduce regresses 512px backward
  (`722.5ms`), and the forward/prep timings move enough to treat this as a
  diagnostic candidate rather than first-class proof. The native-prep follow-up
  validates the next handoff step: `logit_handoff_reduce_vec4_native_prep`
  computes linear sigmoid-MSE prep in Metal and cuts matched 512px prep
  `413.64 -> 37.29ms`, prep+backward `826.35 -> 428.98ms`, and total
  `1446.53 -> 1108.50ms` with F4/F32 parity and zero overflow, but it remains a
  benchmark-only linear colorizer gate. The hidden sigmoid-MSE native follow-up
  passes F4/F32 parity and zero overflow, but rejects naive dense hidden fusion
  as the speed path: H32 scalar totals `317.54/610.90/2549.39ms` at
  `128/256/512px`, H64 256px totals `817.27ms`, and vec4 reduce is slower than
  scalar. The sparse hidden cached-bin native gate is the first positive sparse
  boundary port: at 64f/512px/8192t/F32 it drops H32 sparse64 total
	  `29.66 -> 18.47ms`, H32 sparse128 `111.17 -> 64.17ms`, and H64 sparse64
	  `45.09 -> 28.40ms`, with parity and zero overflow; compare it to the selected
	  sparse-forward batched target/probe route before trainer promotion. The
	  native target-area hidden64 gate is the positive full-support visual-VJP port:
	  matched star-only trainer time drops `5801.7 -> 3496.0ms/step`, and 512px
	  native-only synthetic support survives where the Torch hidden-VJP baseline
	  OOMs, but dense RGB remains `5.648`, so it is speed/memory evidence rather
	  than a quality route. The hidden32 native follow-up cuts mean step to
	  `2464.6ms` but fails the trainer gate (`19.481` probe PSNR), so decoder
	  shrinkage is not the keeper recompute lever. The first
	  real-video
  linear RGB-VJP profile then verifies the handoff boundary against a trainer
  checkpoint: the 64f/512px/8192t 1300-step row matches autograd with zero loss
  error and `9.43e-09` max gradient error, with a small timing win
  (`1691.0 -> 1587.4ms`, `1.065x`) and zero overflow; the 8f/64px smoke is
  cleaner (`78.8 -> 34.7ms`, `2.27x`). This proves the logit-handoff reducer
  can be trainer-compatible for linear RGB reconstruction, but it does not
  cover target-grid V-JEPA MSE or the hidden64 frozen-probe objective. The
  target-grid/frozen-probe VJP bridge profile then checks the current keeper
  objective directly: the 64f/512px/8192t 1300-checkpoint row matches autograd
  with zero loss error and zero overflow. The first autograd-image bridge is a
  slight negative (`1545.5ms` autograd versus `1594.3ms` bridge, `2.57e-08` max
  gradient error), but the analytic target-grid/probe VJP follow-up is
  repeat-positive (`1510.6 -> 1477.2ms`, `1.023x`, `3.07e-08` max gradient
  error). The trainer gate is now wired as
  `feature_target.image_vjp_mode=analytic` and passes the matched 5-step 64f/512
  smoke, but it ties end-to-end step time rather than clearly winning:
  autograd mean step `1303.6ms` versus warm analytic rerun `1304.6ms`
  (`1264.1ms` versus `1259.2ms` after dropping first step). The backward bucket
  improves by `103.3ms`, but manual VJP work moves into the loss bucket. Keep it
  diagnostic. The follow-up sparse-pixel gate is the first target-grid current
  objective speed win that survives the trainer: the repeat-3 profile keeps
  parity (`4.61e-08` max grad error) and drops dense analytic bridge total
  `1245.9ms -> 920.5ms` by replacing dense renderer backward
  (`557.6ms`) with sparse-pixel renderer backward (`46.3ms`) plus
  `184.0ms` sparse packing. The matched 5-step trainer mode
  `feature_target.image_vjp_mode=analytic_sparse_pixels` passes from the
  1300-step checkpoint and cuts no-first step `1318.0ms -> 973.7ms` with only
  `65,536` sparse pixels per 64f/512 step (`0.390625%` of dense). The direct
  sparse-grid follow-up now supersedes it for this keeper diagnostic:
  `feature_target.image_vjp_mode=analytic_sparse_grid` analytically maps the
  trilinear target-grid/probe gradient to sparse source pixels without
  materializing/scanning the dense image gradient, passes profile parity at
  `4.60e-08` max grad error, cuts bridge total to `760.6ms`, and cuts matched
  trainer no-first step to `795.3ms` with `88.6ms` no-first backward. The
  sparse-grid render-mode matrix keeps `feature_direct_gradcache_reduce_vec4`
  as the selected renderer and reports the best checked no-first row at
  `730.5ms` (`78.3ms` mean backward), ahead of gradcache `759.4ms` and direct
  atomic `779.3ms`. The sparse-forward follow-up is the selected current
  diagnostic: `feature_target.image_vjp_mode=analytic_sparse_grid_forward`
  renders only the same `65,536` support pixels and matches dense feature/alpha
  values exactly. The first isolated row cut forward render `515.9ms -> 70.5ms`
  (`7.322x`) and trainer no-first to `492.3ms`, but the 128/256/512 scale
  matrix and isolated repeat show timing is run-order sensitive: all rows pass
  with zero overflow, sequential no-first is `379.2ms`/`494.2ms`/`973.0ms`, and
  the 512px isolated repeat after scale is `598.2ms` no-first / `477.6ms` last.
  The dedicated repeat-3 512px timing gate passes all rows with zero overflow
  and gives no-first step mean/min/max/stdev `504.9/411.0/626.4/110.3ms`,
  last-step `468.8/409.3/549.9/72.7ms`, and no-first backward
  `142.2/114.7/174.4/30.1ms`.
  The batched target-grid/probe VJP path is now a first-class opt-in trainer
  mode: isolated target/probe loss+VJP drops `38.0ms -> 4.8ms` (`7.99x`) with
  `7.45e-09` loss error, the 5-step optimizer harness reaches `173.1ms`
  no-first step with zero overflow, and the integrated trainer repeat-3 gate
  gives no-first step mean/min/max/stdev `179.3/159.7/215.6/31.5ms` with
  zero overflow. The helper-launched 100-step media gate also passes, writes a
  1400-step checkpoint plus RGB-probe contact/MP4 media, and cuts the old
  same-checkpoint target-grid row from `1690.2ms` to `399.9ms` mean step with
  essentially identical objective movement. True native GPU target-grid/probe
  loss+VJP and scalar fixedbin/tile-slot renderer work remain lower-level speed
  targets, but they now need to beat the batched trainer repeat/100-step
  distribution. Visual quality remains the open training gate because the
  contact sheet is still blurry.
  The new
  `firstclass_backward_breakdown.py` split shows the first-class 512px problem
  is not renderer-only: `FeatureToColor`/loss backward is `77.9-83.1%` of
  backward on the 4096/8192t rows, while renderer backward is `16.9-22.1%`
  (`~36%` at 256px/32768t/cap256). The next real speed target is therefore
  optimized fixedbin/tile-slot feature-gradient accumulation plus an
  image-space colorizer/loss VJP or handoff that avoids dense F32 image-gradient
  backprop. The no-pre-norm A/B is the first big whole-graph speed lever:
  512px/8192t/chunk2 with `colorize.pre_norm=false` passes the 2-step trainer
  gate at `3.72s/step` and `1.59s` backward, versus `7.94s/step` and `4.88s`
  backward for the default pre-norm row. The follow-up 20-step media gate also
  passes with zero overflow and media for both variants: no-pre-norm is faster
  (`7.37s/step`, `3.37s` backward versus `11.10s/step`, `7.07s` backward), but
  default pre-norm has slightly better 20-step loss/PSNR (`0.31742` / `4.984`
  versus `0.32053` / `4.941`), so no-pre-norm remains a speed candidate, not a
  promoted quality default. Removing sigmoid too is faster but worse: the
  identity/no-pre-norm diagnostic is the fastest 512px feature row at
  `2.54s/step` and `1.17s` backward, but ends at only `0.32446` loss /
  `4.888` PSNR. Treat this as evidence that the easy decoder simplification is
  speed-only, not the feature quality fix. A hidden-64 pre-norm decoder barely
  improves feature quality (`4.987` PSNR versus `4.984`) while slowing to
  `19.18s/step` and `13.77s` backward, so naive decoder-capacity expansion is
  also not the practical fix. Reducing pre-norm sigmoid init gain from `4` to
  `2` similarly reaches `4.987` PSNR but slows to `14.12s/step` and `8.91s`
  backward. A fresh same-session no-pre-norm 20-step media rerun selects
  `feature_direct_gradcache_reduce_vec4` as the current fast feature diagnostic:
  it matches gradcache loss/PSNR (`0.32053` / `4.941`) and zero overflow while
  improving `2.858s/step`, `1.327s` backward to `2.491s/step`, `1.184s`
  backward. The older RGB-target helper command is
  `src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh
  star-feature-512-rgbfast`. The same helper now also exposes
  `star-feature-512-visual` for compact
  target-area visual overfit (`930.6ms`, `6.023` full RGB on the current-build
  gate) and `star-feature-512-native-fullcell` for the
  promoted exact native vec4 W^T full-support baseline. Compact native
  star-only is rejected because it freezes the colorizer and is slower than
  compact autograd. The dynamic-gsplat fixed-512 comparator now has both a
  5-step smoke and a stronger 20-step media gate at the same
  `64f/512px/8192` active primitive scale. The 20-step gate records
  `2.940s` mean timed step / `1.926s` backward and final eval PSNR `5.587`
  with smeared media, so dynamic gsplat is not the current fast local route or
  quality escape hatch. Report:
  `outputs/benchmarks/2026-05-20_dynamic_gsplat_fixed512_20step_matched_media.md`.
  The selected visual-quality gate explicitly fails scale-up: dense full RGB is
  only `6.023` PSNR, media is sparse/streaked or blurry, and the RGB STAR
  same-clip bracket is `12.444` PSNR. Report:
  `outputs/benchmarks/2026-05-20_star_uvt_selected_visual_quality_gate.md`.
  A trainable low-frequency RGB-grid bridge was added and passes mechanics at
  `353.1ms` mean step / `289.9ms` no-first, but it is also rejected for quality:
  dense full RGB is `5.657` PSNR and feature loss worsens even though grid/probe
  PSNR improves. Report:
  `outputs/benchmarks/2026-05-20_star_uvt_rgb_grid_lowfreq_bridge_gate.md`.
  The selected-shader
  scale gate bounds that result:
  at 256px vec4 is only a small first-class win (`1.112s -> 1.069s` step), and
  at 128px it is a tie/slight backward loss. The same 128px check exposed a
  low-resolution support hazard: 8192 tubes overflow at cap128/default alpha,
  cap256/default alpha, and cap256/`alpha>=1/72`; the first valid 128px row
  needed cap256 plus `alpha>=1/32`. This remains a speed diagnostic, not the
  quality row.
  The precomputed V-JEPA bridge audit originally made the target-representation
  gap explicit: the old `star-feature-512-fast` config had no `features`
  section and was RGB-target `FeatureToColor` training. That is now superseded:
  `star-feature-512-fast` is the cached V-JEPA target-grid batched path, while
  `star-feature-512-rgbfast` keeps the old RGB-target helper. Compare by
  objective and target representation, not by helper name.
  That first bridge smoke now passes: the opt-in
  `feature_target.enabled=true` route loads `rgb_pyramid` through
  `VideoFeatureCache`, adapts `rgb_x1` from `[1,3,8,64,64]` to `[8,32,64,64]`
  with `repeat_truncate`, and trains directly on `render.feature_image`. The
  cache-hit rerun passes with loss `0.34006 -> 0.24809`, zero overflow, all
  model gradients present, `colorizer_grad_required=false`, and
  `93.5ms/step`. The real V-JEPA target smoke now passes too:
  `vjepa_tokens` source shape `[1,1024,768]`, explicit token grid `[4,16,16]`,
  adapted target `[8,32,64,64]`, loss `1.00082 -> 0.90042`, zero overflow,
  `181.1ms/step`, and model-gradient flow present. The chunked 64f/512px real
  V-JEPA target scale gate now passes too: `[1,8192,768]` tokens become a
  channel-adapted `[32,32,16,16]` source and logical `[64,32,512,512]` target,
  with loss `1.000014 -> 0.999545`, zero overflow, `3.743s/step`,
  `1.077s` backward, and `1.734s` target chunk/loss. Channel adaptation now
  runs before grid upsampling after the first 512px attempt exposed a 48 GiB
  interpolation temporary; chunking avoids keeping the full dense target
  resident. The cached-target-layout follow-up then precomputes those adapted
  chunks once (`cached_chunks`, 32 chunks, `2048MiB`, `2.044s` load/prep) and
  cuts the same cache-hit 5-step gate to `1.655s/step`, `0.770s` backward,
  `0.601s` render, and `0.202s` target/loss with the same loss curve and zero
  overflow. The cache budget says this becomes `4GiB` at 128f/512px/F32 or
  64f/512px/F64 and `8GiB` at 64f/1024px/F32. The target-grid follow-up keeps
  only the channel-adapted `[32,32,16,16]` V-JEPA grid resident (`1.0MiB`) and
  downsamples rendered feature chunks before loss; it passes at
  `1.351s/step`, `0.705s` backward, `0.548s` render, and `0.041s` target/loss
  with loss `0.999935 -> 0.999467`. The 20-step target-grid media follow-up
  also passes with monotonic feature loss (`0.999935 -> 0.997425`) at
  `1.451s/step`, `0.722s` backward, `0.630s` render, and `0.037s` target/loss,
  but RGB PSNR/media are not quality evidence because `rgb_loss_weight=0` and
  the colorizer is not trained. The RGB-aux1 target-grid probe trains the
  colorizer and decreases both component losses, but only moves RGB PSNR
  `4.709 -> 4.746` in 20 steps and slows to `2.000s/step` (`1.114s`
  backward). RGB-aux10 barely improves RGB PSNR over aux1 (`4.750`) and slightly
  worsens feature loss, so weight alone is not the missing visual lever. The
  100-step aux10 row moves more clearly (`RGB PSNR 4.709 -> 5.109`, feature
  loss `0.999935 -> 0.964670`) at `1.876s/step`, which says schedule length
  matters. A matched RGB-warm20 schedule (`feature=0/rgb=20` for 20 steps,
  then `feature=1/rgb=10`) is faster at `1.639s/step` but ends worse
  (`RGB PSNR 5.046`, feature loss `0.973557`), so early feature-loss skipping
  is not the visual fix. The standalone target-grid feature-to-RGB probe now
  proves the cached target features are decodable: hidden64 `FeatureToColor`
  reaches grid PSNR `23.401` and full upsampled PSNR `20.073` in `2.427ms/step`
  from the same `[32,32,16,16]` V-JEPA target grid. This is cached-feature
  speed/memory evidence and an oracle decoder check, not a source-view quality
	  baseline. The frozen-probe STAR integration gate now passes too:
	  `1.220s/step`, zero overflow, feature loss `0.999935 -> 0.998357`, and probe
	  PSNR `13.985 -> 14.060`. That proves cheap wiring and gradient flow, but the
	  short-run visual gain is tiny. The matched 100-step frozen-probe row moves
	  more clearly at similar cost: `1.268s/step`, feature loss
	  `0.999935 -> 0.970035`, and probe PSNR `13.985 -> 14.641`. It is the better
	  visual diagnostic and cheaper than 100-step RGB-aux10. The 300-step extension
	  keeps moving at `1.355s/step`, feature loss `0.999935 -> 0.811652`, and probe
	  PSNR `13.985 -> 16.560`. It still trails the standalone `20.073` PSNR oracle;
	  the checkpoint/no-media rerun matches that curve at `1.268s/step`, and the
	  resumed 300-step continuation reaches feature loss `0.655366` and probe PSNR
	  `19.884` at `1.440s/step`. That nearly reaches the standalone full-video
	  upsample number (`20.073`). A probe-emphasis 600->800 continuation reaches
	  probe PSNR `21.425` at `1.512s/step` with zero overflow, but feature loss
	  drifts `0.655132 -> 0.703820`. A scheduled 800->1000 balance continuation
	  recovers feature loss `0.703862 -> 0.643852` at `1.308s/step`, but gives
	  back a little probe PSNR (`21.428 -> 21.382`) and is nonpassing on
	  probe-loss decrease. A feature0.5/probe40 1000->1100 Pareto continuation
	  passes the combined gate at `1.461s/step`, moves probe PSNR
	  `21.384 -> 21.789`, and keeps zero overflow, but feature loss drifts
	  `0.643823 -> 0.656728`. A 1100->1200 recover schedule lowers feature loss
	  `0.656765 -> 0.635093` at `1.521s/step`, but gives back a little probe PSNR
	  (`21.792 -> 21.738`) and is nonpassing. A short feature0.75/probe40
	  1200->1250 continuation passes at `1.523s/step` and restores probe PSNR
	  `21.740 -> 21.929`, but feature loss rises `0.635066 -> 0.638799`, so
	  a feature1/probe40 1250->1300 continuation was run next and is the first
	  current both-improving row: feature loss `0.638803 -> 0.632192`, probe
	  PSNR `21.933 -> 21.963`, zero overflow, and `1.285s/step`. The target-route
	  extension to 1400 keeps both improving: feature loss `0.632124 -> 0.627129`,
	  probe PSNR `21.965 -> 21.979`, zero overflow, and `1.690s/step` on the
	  older dense target-grid path. The sparse-forward batched-VJP helper/media
	  row preserves that movement at `399.9ms/step` mean and `262.9ms/step`
	  last-20. The effective-lr001 sparse-forward rerun keeps the dense lr001
	  endpoint at `372.3ms/step` mean, `158.9ms` backward, feature loss
	  `0.630549`, and probe PSNR `22.034`, but gives up lr005's better feature
	  loss and has noisy late timing. The target-route risk is now closing the
	  same-grid oracle (`23.401`) or beating the batched speed surface with
	  native VJP. A matched
	  timing repeat reproduces the dense slowdown at `1.711s/step` with zero
	  overflow and `68/45/128` max/p95/cap tile count, so do not blame tile
	  overflow first; the continuation-chain report is
	  `outputs/benchmarks/2026-05-19_star_uvt_feature1_continuation_chain.md`.
	  The new whole-graph profile report
	  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_wholegraph_profile.md`
	  shows the current target-grid/frozen-probe objective is renderer-backward
	  dominated (`81.3-81.4%` of manual backward), but the isolated manual split
	  does not reproduce the end-to-end 1300-source slowdown (`1565.9ms` at
	  step 1250 vs `1504.0ms` at step 1300). Treat the remaining speed question
	  as trainer-autograd/MPS trace variance or native-VJP work, not tile
	  overflow. The follow-up trainer trace adds `step_timings_ms` to trainer
	  JSON output and reproduces the 1300-source slowdown after dropping the
	  first optimizer/warmup step (`1850.7ms` vs `1705.3ms`), with a late
	  objective spike at global step `1318`; report:
	  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_trainer_trace_report.md`.
	  The chunk trace report
	  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_chunktrace_report.md`
	  localizes that spike as distributed rather than a single bad chunk:
	  `27/32` chunks worsen from 1317 to 1318, the first quarter contributes
	  `44.5%` of the weighted-loss jump, and the elevated loss persists at
	  1319. The optimizer/LR checkpoint gate then confirms schedule sensitivity:
	  the original `lr=0.005` optimizer continuation fails, while the corrected
	  retained-optimizer `lr=0.001` continuation records checkpoint/effective
	  LRs `[0.005] -> [0.001]`, removes the 1318 spike, and passes with end loss
	  `0.884576`, feature loss `0.631648`, probe PSNR `21.991`, no-first
	  `1384.4ms/step`, and `748.9ms` backward. The reset-optimizer `lr=0.001`
	  control also passes (`0.884902`, `0.631614`, `21.984`) but is slower in
	  this diagnostic. Report:
	  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_reset_report.md`.
		  Effective `lr=0.001` from the 1300 checkpoint is the safer probe/visual
		  continuation path; any new native VJP/scalar fixedbin speed work should
		  beat the sparse-forward batched-VJP helper before replacing it. The
	  100-step effective-lr001 continuation from 1300 passes with media/checkpoint
	  and reaches feature loss `0.630549`, probe PSNR `22.034`, mean
	  `1463.8ms/step`, and `778.4ms` backward. It avoids the early 1318 jump but
	  later has a smaller transient at `1377->1378`; the older lr005 1300->1400
	  row is slower and lower on probe PSNR, but better on final feature loss
	  (`0.627129`) and slightly better weighted loss (`0.880751`). Report:
	  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr001_100step_report.md`.
	  The matched effective-lr001 sparse-forward rerun preserves that dense
	  lr001 endpoint at `372.3ms/step` mean and `158.9ms` backward, but it keeps
	  the same quality tradeoff and noisy late timing. Report:
	  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_lr001_64f512_from1300_100step_media.md`.
	  The follow-up checkpoint-selection gate chooses the lr005-sparse 1400 state
	  for further quality work: it passes 50 effective-lr001 steps to feature
	  loss `0.625976` and probe PSNR `22.010`, while lr001-sparse 1400 fails
	  after a `1444 -> 1445` jump and ends at feature loss `0.631770` / probe
	  PSNR `21.843`. Report:
	  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_checkpoint_select_1400_to1450.md`.
	  The selected lr005-sparse 1450->1500 media gate also passes and writes
	  the next checkpoint/media: loss `0.877762 -> 0.876224`, feature loss
	  `0.625962 -> 0.625428`, probe PSNR `22.010 -> 22.027`, mean
		  `315.8ms/step`, last-20 `254.0ms/step`, zero overflow, but the contact
		  sheet remains blurry. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_lr005sparse_1450_to1500_media.md`.
		  The full-resolution autograd RGB-aux probe-init bridge from that sparse
		  1500 checkpoint is a negative quality result: RGB loss improves
		  `0.272626 -> 0.259968`, but feature loss worsens
		  `0.625418 -> 0.626799`, frozen-probe PSNR drops `22.028 -> 21.879`,
		  trainable-colorizer media artifacts appear, and mean step time jumps to
		  `5206.6ms` (`16.5x` slower than sparse 1500). Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_autograd_rgbaux_probeinit_from1500_negative.md`.
		  The rendered-feature sparse-pixel RGB probe from the same sparse 1500
		  checkpoint trains only a hidden64 colorizer on actual rendered sparse
		  pixels and passes the sampled loss gate (`0.168261 -> 0.099014`,
		  sparse PSNR `7.740 -> 10.043`) at `241.4ms/step`, but dense full-video
		  PSNR is only `6.096` and media remains sparse-streaked. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels.md`.
		  The denser stratified64 rendered-pixel follow-up samples `262,144`
		  full-resolution pixels/step (`4x` the previous rendered-feature probe)
		  and still only reaches `6.132` dense full-video PSNR at `331.5ms/step`.
		  Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_stratified64.md`.
		  The first native sparse visual VJP gate updates STAR parameters from sparse
		  RGB loss (`model_grad_seen=true`) at `336.8ms/step`, but the frozen
		  target-grid colorizer lands at only `5.739` full-video PSNR. Report:
		  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe.md`.
		  The joint sparse visual VJP follow-up trains STAR and the hidden64
			  colorizer together (`model_grad_seen=true`, `colorizer_grad_seen=true`)
			  and improves full-video PSNR to `6.025`, but it still trails the
			  colorizer-only stratified diagnostic (`6.132`) and costs
			  `729.4ms/step`. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe.md`.
			  The mixed target-grid/probe plus sparse visual VJP follow-up preserves
			  feature/probe movement and raises sparse visual sample PSNR to `6.036`,
			  but dense full-video PSNR remains `6.024` while step time rises to
			  `964.0ms`; this is a mechanics pass, not a quality promotion. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_mix_from1500.md`.
			  The patch2x2 same-pixel support follow-up is faster (`619.5ms/step`)
			  and raises sparse visual sample PSNR to `6.179`, but feature-target
			  loss worsens and dense full-video PSNR drops to `6.000`; this rejects
			  contiguous sparse patch support as the visual fix. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patch2x2_from1500.md`.
			  The patch-mean64 visual-basis follow-up samples `1,048,576` sparse
			  visual pixels/step and pools them into `262,144` local-mean cells. It
			  passes and restores feature/probe movement with dense full-video PSNR
			  `6.023`, but costs `1124.6ms/step` and still has sparse/high-frequency
			  media. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patchmean64_from1500.md`.
			  The target-area64 follow-up keeps the same sparse visual support
			  but compares against true area-downsampled RGB target cells. It is
			  slightly faster (`1103.1ms/step`) and raises sparse visual PSNR to
			  `6.064`, but dense full-video PSNR remains `6.023` and media is
			  unchanged. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_from1500.md`.
			  The phased target-area64 follow-up cycles the compact `2x2`
			  support through a `4x4` subcell schedule. It passes and raises
			  sparse visual PSNR to `6.077`, but dense full-video PSNR falls to
			  `6.019` at `1169.2ms/step`; fixed support position is not the
			  quality blocker. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_phase2x2_from1500.md`.
			  The full-cell8 target-area follow-up sends gradients through every
			  pixel in each `8x8` target-area cell (`16,777,216` visual pixels/step
			  into `262,144` loss cells). It is nonpassing: feature loss and probe
			  PSNR worsen, dense full-video PSNR falls to `5.722`, and mean step is
			  `7526.7ms` with `5702.6ms` in sparse visual loss construction. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_from1500.md`.
			  The manual hidden64 VJP version matches that endpoint while cutting
			  sparse visual loss construction to `3803.6ms` and mean step to
			  `6414.0ms`, but it remains nonpassing and quality-negative. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_manualvjp_from1500.md`.
			  The star-only manual hidden64 variant skips colorizer parameter
			  gradients and cuts mean step to `5801.7ms`, but dense RGB drops to
			  `5.648`, so it is only a lower-bound diagnostic. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_staronly_from1500.md`.
			  The fast-GELU derivative variant keeps colorizer gradients but is
			  rejected: mean step is `6252.1ms`, dense RGB stays `5.722`, and the
			  profiled loss-side total is worse than exact manual. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_fastgelu_from1500.md`.
			  The compact linear colorizer/manual VJP variant cuts the full-cell8
			  row to `2064.4ms/step` with `383.3ms` sparse visual loss construction,
			  but the linear probe is much weaker than hidden64 (`16.980` full PSNR)
			  and dense RGB remains only `5.668`; keep it as a mechanics diagnostic,
			  not a promoted route. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_linear_from1500.md`.
			  The hidden32 manual VJP follow-up keeps most hidden64 target-grid
			  probe capacity (`19.704` full PSNR vs `20.073`) but remains too slow
			  for dense full-cell8 Python/Torch loss VJP (`4298.4ms/step`,
				  `2136.1ms` sparse visual loss construction) and dense RGB remains
				  `5.678`; it is a Pareto diagnostic, not the route. Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_hidden32_from1500.md`.
				  The native target-area hidden64 follow-up is the first positive
				  full-support native gate: it passes parity, wins synthetic
				  full-support timing at 128/256px, survives 512px native-only where
				  Torch hidden VJP OOMs, and cuts the matched star-only trainer row
				  `5801.7 -> 3496.0ms/step` while keeping the same `5.648` dense RGB
				  endpoint. Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_native_gate.md`.
				  The native hidden32 target-area follow-up is faster (`2464.6ms/step`,
					  `1321.7ms` sparse visual backward) but fails the gate with probe PSNR
					  `19.481` and full RGB `5.632`; reject decoder shrinkage as the
					  recompute fix. Report:
					  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden32_gate.md`.
					  The benchmark-only skip-feature-grad diagnostic proves raw feature
					  atomics are not the hidden64 native bottleneck: backward only improves
					  `594.9 -> 562.2ms` at 256px and `1918.6 -> 1854.3ms` at 512px.
					  Report:
					  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_skip_feature_grad_gate.md`.
					  The opposite benchmark-only feature-only/geometry-only split
					  confirms the bottleneck is shared hidden64 recompute/traversal:
					  same-session full/feature-only/geometry-only backward is
					  `581.3/548.2/547.3ms` at 256px and
					  `1919.7/2106.7/2174.0ms` at 512px. Report:
					  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_geometrysplit_gate.md`.
					  The recompute-only mode disables all output-gradient atomics and
					  still costs `571.3ms` backward at 256px and `2101.7ms` at
					  512px, so the shared replay/hidden64 VJP envelope is the floor.
					  Report:
					  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_recompute_floor_gate.md`.
					  Traversal-only skips hidden64 VJP too and drops backward to
					  `194.9ms` at 256px and `742.2ms` at 512px, isolating the
					  hidden64 forward/VJP slice as the largest removable piece.
					  Report:
					  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_traversal_floor_gate.md`.
					  Hidden-forward-only splits that hidden slice into forward
					  `150.6/450.6ms` and backward `225.8/909.0ms` at 256/512px,
					  making W^T/GELU feature VJP the larger hidden subtarget.
					  Report:
					  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden_forward_backward_split_gate.md`.
					  Hidden-preact-only narrows that again: output+GELU prebackward
					  is only `54.8/61.7ms`, while the F32 W^T feature-gradient
					  matvec is `171.0/847.3ms`. Report:
					  `outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden_preact_wt_split_gate.md`.
					  The row-major W^T follow-up keeps exact gradients but rejects
					  simple W^T loop reordering as a trainable speed path: full
					  native backward slows `647.4 -> 711.5ms` at 256px and
					  `2040.5 -> 2161.6ms` at 512px; recompute-only barely improves
					  `572.1 -> 555.8ms` and `1993.0 -> 1983.4ms`. Report:
					  `outputs/benchmarks/2026-05-20_star_uvt_native_target_area_rowmajor_wt_gate.md`.
					  The vec4 W^T follow-up is the first positive exact W^T kernel
					  reduction: same-build full backward improves `675.9 -> 642.2ms`
					  at 256px and `2408.1 -> 1804.7ms` at 512px, with repeat
					  `1832.8ms`; recompute-only improves `586.6 -> 518.3ms` and
					  `2305.2 -> 1635.8ms`. The current-build trainer A/B promotes
					  vec4 W^T for full-support native target-area star-only:
					  mean step `4262.1 -> 4071.0ms`, mean backward
					  `3700.2 -> 3152.6ms`, mean sparse visual backward
					  `2546.7 -> 1963.5ms`, matched endpoint class. Reports:
					  `outputs/benchmarks/2026-05-20_star_uvt_native_target_area_vec4_wt_gate.md`.
					  `outputs/benchmarks/2026-05-20_star_uvt_native_target_area_vec4_wt_trainer_ab_gate.md`.
					  The 50-step promoted-mode gate passes at `3359.2ms` mean
					  step / `3072.1ms` last step, full RGB `5.732`, and zero
					  overflow, but remains slower and lower quality than the
					  compact target-area64 helper route on the fresh current-build
					  gate (`930.6ms`, `6.023` RGB).
					  Report:
					  `outputs/benchmarks/2026-05-20_star_uvt_native_target_area_vec4_wt_50step_gate.md`.
					  The compact helper route is therefore the practical visual
					  overfit route; full-cell8 native vec4 W^T is the exact
					  full-support baseline. Report:
					  `outputs/benchmarks/2026-05-20_star_uvt_compact_target_area_visual_route_gate.md`.
					  Compact native star-only is rejected: it passes mechanically
					  but freezes the colorizer and costs `2265.0ms` mean step.
					  Report:
					  `outputs/benchmarks/2026-05-20_star_uvt_compact_native_staronly_diagnostic.md`.
					  Compact manual hidden64 preserves colorizer gradients, but
					  is also rejected: first-five mean/no-first step is
					  `2007.4/1899.2ms`, feature/probe quality regresses, and it
					  trails compact autograd's `991.9/787.7ms` comparison row.
					  Report:
					  `outputs/benchmarks/2026-05-20_star_uvt_compact_visual_vjp_gate.md`.
					  Native colorizer-gradient vec4 W^T closes the missing
					  returned-gradient ABI and passes tiny STAR/colorizer parity,
					  but fails the trainer gate at `2738.7ms` mean step and
					  `1474.2ms` backward with the same feature/probe regression.
					  Report:
					  `outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_vec4_wt_tiny_gate.json`.
					  The follow-up colorizer-gradient-only split pins the compact
					  native failure on parameter-gradient atomics: direct compact
					  backward is `88.9ms` star-only, `536.6ms` colorizer-only,
					  and `531.4ms` full colorizer. Report:
					  `outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_atomic_split_gate.md`.
					  The Torch sidecar reducer prototype is correct but rejected:
					  it improves over native atomics (`390.9ms` vs `752.8ms`)
					  but loses to the sparse-pixel baseline (`276.6ms`) due to
					  duplicate sparse render/hidden replay. Report:
					  `outputs/benchmarks/2026-05-20_star_uvt_native_target_area_torch_reducer_prototype_gate.md`.
					  The same-pass SIMD-reduced colorizer follow-up fixes the
					  direct atomic envelope (`297.2ms` native compact total vs
					  `312.1ms` sparse-pixel baseline in the matched run), but
					  the 5-step trainer still rejects it: `2908.9ms` mean step,
					  `1363.0ms` backward, `604.0ms` sparse visual backward, and
					  the same feature/probe regression. Report:
					  `outputs/benchmarks/2026-05-20_star_uvt_native_target_area_colorizer_simdreduce_gate.md`.
					  The matched 512px native handoff gate fills the scale gap for
				  direct-kernel RGB/loss handoff prototypes: `fused_first3` passes at
			  `494.09ms` backward / `1152.58ms` total, `linear_sigmoid_mse` is
			  rejected at `918.09ms` backward, and `logit_handoff_reduce_vec4`
			  shows the best native backward (`386.26ms`) but still pays
			  `421.89ms` Torch prep. Report:
			  `outputs/benchmarks/2026-05-19_star_uvt_native_handoff_matched_512_gate.md`.
				  The hidden sigmoid-MSE native gate passes parity but is not the
				  next keeper: H32 scalar is `610.90ms` total at 256px and
				  `2549.39ms` at 512px, H64 256px is `817.27ms`, and vec4 reduce is
				  slower than scalar. Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_hidden_sigmoid_mse_native_gate.md`.
				  The sparse hidden sigmoid-MSE cached-bin native gate is positive
				  at the sparse visual boundary: H32 512px sparse64 total drops
				  `29.66 -> 18.47ms`, H32 sparse128 drops `111.17 -> 64.17ms`,
				  and H64 sparse64 drops `45.09 -> 28.40ms`, all with parity and
				  zero overflow. Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_sigmoid_mse_native_gate.md`.
				  The first trainer-wired native hidden pixel64 gate is correct
				  but not faster: warm sparse loss+backward is `113.25ms`
				  manual versus `116.27ms` native, with final sparse loss
				  matching within `3.26e-08`. Report:
				  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_nativehidden_trainer_gate.md`.
					  The split manual-VJP subphase profiles show exact GELU backward
			  (`~1.34-1.44s`) and fc1 (`~0.75-0.89s`) dominate the remaining
			  loss-side cost; target-area reduction is only `~0.12-0.13s`.
			  Reports:
			  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_subphase_profile_split_fullstep.md`
			  and
			  `outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_staronly_subphase_profile_fullstep.md`.
		  The first explicit optimizer-LR schedule gate (`0.001` until global step
		  `1375`, then `0.00025`) is negative for promotion: it removes the
	  `1377->1378` jump, but moves a comparable jump to `1385->1386`; the
	  100-step scheduled row ends worse than static lr001 on weighted loss
	  (`0.881602` vs `0.880942`), feature loss (`0.630803` vs `0.630549`),
	  probe PSNR (`22.027` vs `22.034`), and timing (`1506.9ms` / `807.2ms`
	  backward vs `1463.8ms` / `778.4ms`). The diagnostic 88-step late trace is
	  expected to fail the quality pass bit because it stops just after the
	  spike; it confirms the spike is distributed (`26/32` chunks worsen, sum
	  `0.015248`, max frame `0` chunk `0.001802`). Report:
	  `outputs/benchmarks/2026-05-19_star_uvt_feature1_probe40_lr_schedule_report.md`.
		  Quality now needs checkpoint selection or a schedule keyed to transient
		  recovery; speed work should target a native path only if it beats the
		  sparse-forward batched-VJP helper.
	  The feature overfit trainer now supports opt-in warm-start checkpoint/resume with optimizer state,
	  explicit `train.global_step_offset`, and smokes on both the 8f/64px
	  RGB-pyramid route and the real 64f/512px frozen-probe route, so longer
	  probes no longer need to restart from initialization.
  The normalized V-JEPA comparison report now reads that scale gate against the
  existing Gaussian/token artifacts: STAR V-JEPA streaming target is
  `3.743s/step`, STAR V-JEPA cached-chunks target is `1.655s/step`, the
	  STAR V-JEPA target-grid diagnostic is `1.351s/step` (`1.451s/step` for the
	  20-step media row, about `2.000s/step` with 20-step RGB aux, `1.876s/step`
	  for 100-step aux10, `1.639s/step` for the negative RGB-warm20 row,
	  `0.00243s/step` for the standalone target-grid feature-to-RGB oracle, and
	  `1.220s/step` / `1.268s/step` / `1.355s/step` / `1.440s/step` for the
	  20/100/300/resume300 integrated frozen-probe rows, plus `1.512s/step`
	  for probe-emphasis, `1.308s/step` for scheduled balance, and
	  `1.461s/step` for feature0.5/probe40, `1.521s/step` for recover
	  schedule, `1.523s/step` for feature0.75/probe40, and `1.285s/step` for
	  feature1/probe40 plus `1.690s/step` / `1.711s/step` for the dense
	  1300->1400 extension and timing repeat, then `0.400s/step` mean /
	  `0.263s/step` last-20 for the lr005 sparse-forward batched-VJP helper row,
		  and `0.372s/step` mean / `0.539s/step` last-20 for the lr001
		  sparse-forward rerun, plus `0.316s/step` mean / `0.254s/step` last-20
		  for the selected lr005-sparse 1450->1500 media gate, plus `5.207s/step`
			  for the negative autograd RGB-aux probe-init bridge, plus `0.241s/step`
			  for the rendered-feature sparse-pixel RGB probe, `0.332s/step` for
			  the stratified64 rendered-pixel probe, and `0.337s/step` for the sparse
				  visual VJP frozen-probe gate, plus `0.729s/step` for the joint sparse
				  visual VJP gate and `0.964s/step` for the mixed target-grid/probe
				  sparse visual VJP gate, plus `0.620s/step` for the patch2x2 support
				  gate, `1.125s/step` for the patch-mean64 visual-basis gate, and
				  `1.103s/step` for the target-area64 visual-basis gate, and
				  `1.169s/step` for the phased target-area64 visual-basis gate, and
				  `7.527s/step` for the full-cell8 target-area gate, and
				  `6.414s/step` for the manual hidden64 VJP variant, and
					  `5.802s/step` for the star-only manual hidden64 diagnostic, and
					  `6.252s/step` for the fast-GELU manual hidden64 reject, and
					  `2.064s/step` for the compact manual-linear diagnostic, and
					  `3.496s/step` for native full-cell target-area star-only), the
  selected STAR RGB feature diagnostic is
  `2.491s/step`, Gaussian/token 64f/512px/8192 recon-only cached conditioning
  is `3.460s/step`, and the old Gaussian/token
  prediction-side V-JEPA-loss run is still `38.621s/step` with `36.762s`
  backward.
  Gate 4 same-clip quality now fails
  feature promotion: RGB STAR direct-atomic on the same 512px/64f/8192t
  test-video bracket reaches `12.44` PSNR in 20 steps, while the best feature
  row reaches `4.99` PSNR and the fastest feature row is the lower-quality
  identity diagnostic; projected dynamic/F32 rows remain speed-only references in
  `outputs/benchmarks/2026-05-19_star_uvt_gate4_quality_bracket.md`. The 512px
  bracket now passes at 4096 and 8192 tubes with zero overflow under
  `feature_direct_gradcache` (max tile `18` and `33`), but the rows are already
  too slow (`6.46s/step` and `7.94s/step`), so 512px/32768t should wait for a
  backward or colorize/loss speed change. The 8f
  first-class smoke and the promoted 64f rows write
  contact sheets and side-by-side MP4s, and the generated scale report lives
  under `outputs/benchmarks/`. RGB `star_uvt_v0` remains a separate 3-channel
  `float3` path.
- The 300-clip V-JEPA/static-dynamic Gaussian lane is cache-hot with prefetch
  at 256px, but the 512px promotion produced NaNs. Do not treat that multires
  config as a completed scale baseline until the promotion is guarded or fixed.
- Same-view and multicam loaders are documented and reusable. The first mixed
  same-view plus heldout-novel-view trainer/scheduler bridge now exists and
  passes small offline smokes; the next bridge is a longer W&B trace with media
  and separate same-view/heldout trend evidence before any baseline promotion.
- V-JEPA/F32 multicam has real heldout evidence, but benchmark claims still
  need source/camera-disjoint manifests, explicit leakage probes, and
  `BASELINES.md` rows.
- WorldFoam Gate4 shader work is active but narrow. Lean owner-run fused-MSE
  recompute is the current compute keeper, but its tape storage still scales
  almost linearly. Packed endpoint owner-run delta replay is now wired as
  `owner-run-delta-packed-recompute-fused-mse-nomid` and has moving-ray
  loss/site-gradient parity against the lean owner-run path, while the storage
  probe shows exact length recovery, owner/count parity, and `5.76x` storage
  growth over an `8x` frame increase. Treat it as shader research, not full
  system parity; clean `2/4/8/16f` timing is still pending because the first
  promotion attempt was blocked by a contended benchmark environment.
  Factorized coefficient recompute is now a real Metal fork:
  `owner-run-delta-packed-factorized-recompute-fused-mse-nomid` consumes
  `boundary_f32 + track_ray_coeff_f32`, removes resident `delta_coeff_f16`,
  passes moving-ray parity against lean owner-run, and has a contended
  functional render16/site8 `2/4/8/16f` ladder with selected storage scaling
  `1.875x` over `8x` frames and resident coeff storage scaling `1.0x`. Clean
  promotion timing is still pending. Separately, the WorldFoam paper-math lane
  now has `research_notes/worldfoam_paper/WORLD_FOAM_MATH_APPENDIX.md`, whose
  current claim is gauge-covariant optical transfer factored through a compiled
  cell-path atlas. The next paper-math gate is a constant-density owner-run
  cell-path fixture with same-representation replay and exact VJP finite
  differences, now specified at
  `research_notes/worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md`;
  boundary flux, flux witness scores, Hessians, feature-gauge transfer, and
  ray-space transfer stay behind tests. The frame-bitmask
  selector variant is now
  the promoted synthetic selector at site8 and site24/high-cap, and the
  render64/site24 path is correctness-green through 16f after widening the
  frame-bitmask track/change offset prefixes to int32. Do not claim broad STAR
  UVT competitiveness yet: 8f/16f render64 timings were contaminated, and the
  16f artifact shows CPU/slow-owner-run tape prep dominating wall time before
  the GPU step. A selected-only owner-run delta prep flag now skips baseline
  segment-tape accounting for shader timing artifacts. Exact native owner-run
  cutwalk prep now moves endpoint-record sequence construction into C++ with
  midpoint-owner and threshold parity; focused owner-run tests pass, CPU and
  MPS shader-output parity now cover duplicated multiview moving rays/view-major
  sample order, 4f train
  sequence prep is down to `0.553s`, and the native-prep `2/4/8/16f` path
  ladder is `status=ok` with sublinear backward and total step scaling. The
  matched STAR comparison harness now rejects missing or contended WorldFoam and
  STAR artifacts, reserves top-level `worldfoam_artifact` for the clean
  promotable artifact actually passed to STAR, supports `--preflight-only` and
  `--preflight-stability-samples`, and can run the final acceptance audit
  `research_experiments/world_foam_lane2/verify_worldfoam_star_native_cutwalk_promotion.py`
  via `--verify-promotion`. The clean gated promotion finally passed at
  `2026-05-20_native_cutwalk_worldfoam_star_starretry.promotion_summary.json`:
  wrapper `status=ok`, WorldFoam and STAR benchmark environments both
  `background`, one promotable WorldFoam attempt, one promotable STAR attempt,
  and integrated verifier `status=ok`. The selected WorldFoam artifact is
  `2026-05-20_native_cutwalk_worldfoam_star_starretry.attempt1.worldfoam.json`
  and the STAR comparison is
  `2026-05-20_native_cutwalk_worldfoam_star_starretry.star_attempt1.star_compare.json`.
  WorldFoam artifact means are `3.008/3.014/3.323/4.095ms` total and
  `2.739/2.517/2.561/3.796ms` backward for `2/4/8/16f`, with `1.361x` total
  and `1.386x` backward scale over an `8x` frame increase; train PSNR ends
  `11.770/11.783/12.150/12.248` and heldout PSNR ends
  `12.352/12.406/12.589/12.857`. The matched STAR comparison medians are
  `5.003/5.943/8.092/9.794ms` total and `2.629/3.411/5.083/6.768ms` backward,
  so WorldFoam is faster on total and backward at all four frame counts in this
  micro-gate while STAR still has the broader RGB-quality lineage. Do not read
  this as full system parity; it is a clean render64/site24 fused-MSE Gate4
  speed/scale result, now recorded as dated micro-gate rows in `BASELINES.md`.
  A synthetic repeated-fixture 32f extension also passes the strict wrapper and
  verifier after fixing the framebitmask signed-int32 bit-31 boundary:
  `2026-05-20_native_cutwalk_worldfoam_star_repeat32_framebitmask32fix` requests
  `2/4/8/16/32f`, records `loaded_frame_count=16` and repeat metadata for the
  32f rows, and keeps WorldFoam faster than matched STAR at every requested
  frame count. Treat that as speed-scaling smoke only; a real longer-than-16f
  fixture or quality-linked gate is still the next stronger claim. A subsequent
  render96/site48 diagnostic smoke found and fixed the next framebitmask shader
  correctness blocker: base offsets can exceed int16 (`83695` observed), so the
  framebitmask path now keeps `base_offsets_i32` through prep, storage
  accounting, Python binding, C++ dispatch, and Metal shader input. That smoke
  is not promotable timing evidence because the benchmark environment was
  contended. The strict follow-up
  `2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate`
  passed after the wrapper rejected a first ai_trader-contended attempt and
  promoted attempt 2. WorldFoam render96/site48 medians are
  `3.760/4.125/4.619ms` total and `3.480/3.847/4.331ms` backward for
  `2/4/8f`, versus matched STAR 96px/1792-tube medians
  `5.773/7.583/9.692ms` total and `3.614/5.161/6.719ms` backward. This is a
  larger fused-MSE speed/scale gate with i32 framebitmask offsets, not RGB
  system parity. The wrapper/verifier now also has a strict real-frame contract:
  `--worldfoam-config` and `--star-video-path` can point future gates at
  non-default fixtures, and `--require-real-loaded-frames` makes promotion
  verification reject artifacts that used repeated loaded frames or lack
  `loaded_frame_count` metadata. The 16f checked-fixture blocker is now partly
  unblocked by `src/dataset_configs/multicam_val_deepview_03dog_128_8fps_32f.jsonc`
  plus
  `src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_real32_32_smoke.jsonc`:
  a one-step render16/site8 native-cutwalk smoke on real DeepView 32f data
  passed with `loaded_frame_count=32`, no repeat flags, loss decrease, nonzero
  gradient, and parameter update. Its artifact
  `2026-05-20_worldfoam_real32_native_cutwalk_loader_smoke.json` ended
  `benchmark_environment.status=contended` due `MTLCompilerService`, so it is
  a correctness/data gate only. A true 32f speed or STAR-comparison promotion
  still needs a clean-background wrapper run with `--require-real-loaded-frames`.
  Warm real32 retries now prove the shader path itself is fast once compiled:
  `2026-05-20_real32_strict_mini_wrapper_settle_retry` records two clean-start
  32f attempts at about `2.25-2.30ms` total and `1.95-2.01ms` backward, but
  both were correctly rejected as `worldfoam_not_promotable` because the live
  `ai_trader` TOTO export started before the post-run snapshot. No STAR
  comparison ran from that summary. The benchmark preflight now treats that
  periodic TOTO MPS-export monitor as a blocker even while its parent process is
  idle, so the next full wrapper should fail fast until that screen exits or is
  paused. The final promotion verifier now also proves real-input command
  lineage under `require_real_loaded_frames=true`: the summary must record the
  custom WorldFoam config and STAR video path, the stored WorldFoam/STAR
  commands must pass those paths, and the planned/selected STAR commands must
  point `--worldfoam-artifact` at the selected WorldFoam artifact. It also
  verifies artifact lineage by matching WorldFoam `config_path` and STAR
  `star.video_path` back to the recorded inputs. The wrapper now records parsed
  `frame_counts`, and the real-frame verifier checks WorldFoam/STAR artifact
  rows against that exact requested
  frame set. The wrapper now fails early if `--require-real-loaded-frames` is
  requested without both explicit real inputs; tests cover neither-input and
  one-sided-input failures. It also rejects empty, non-integer, nonpositive, and
	  duplicate `--frame-counts` at parse time. The focused lane gate is `79` tests
	  passing plus scoped static checks after the strict environment gate refresh:
	  `unchecked` benchmark-environment probes now block promotion instead of
	  silently behaving as clean, while truly quiet `ok` snapshots promote the same
	  way as `background` snapshots, and the built native packed-extension fixture
	  is included in the gate. The final promotion verifier now requires a non-empty
	  WorldFoam acceptance block instead of letting missing acceptance metadata
	  promote, STAR compare rejects missing acceptance before spending a matched
	  STAR timing run, and the wrapper itself refuses to select a WorldFoam artifact
	  when acceptance metadata is missing. The quality bridge report
	  `2026-05-20_worldfoam_star_quality_bridge_native_cutwalk_microgate.json`
	  now records the current honest stance: WorldFoam is speed-competitive in the
	  clean micro-gate, but not RGB-quality competitive with STAR UVT or even the
	  solid same-source baseline (`12.248` best train PSNR, `12.857` heldout PSNR,
	  `17.575dB` train gap to STAR source). It also includes the existing
	  render96/site48 capacity candidate and records that it does not improve the
	  train PSNR gap (`9.875` best train PSNR) or any overlapping primary frame
	  (`-2.55/-2.53/-2.27dB` train PSNR at `2/4/8f`); it also flags that the
	  candidate is missing the primary `16f` row, so the capacity negative is not
	  silently treated as full-frame-set coverage. The bridge also separates a
	  future quality-closing capacity candidate from a broad STAR claim: if the
	  best quality artifact is not the primary matched-speed artifact, it sets a
	  matched-speed-needed flag instead of inheriting the old speed gate. The
	  train/eval path now exposes
		  `--site-initialization {legacy_sparse,legacy_pixel_mean,legacy_frame_pixel_mean,legacy_frame_patch3_mean,stratified_grid,stratified_pixel_mean}`
		  and records the selected mode in artifacts; default `legacy_sparse`
		  preserves old artifacts. Direct initializer tests now cover
		  `stratified_grid` geometry, `legacy_pixel_mean` global color averaging,
		  `legacy_frame_pixel_mean` frame-local color averaging,
		  `legacy_frame_patch3_mean` same-frame 3x3 patch color averaging, and
		  `stratified_pixel_mean` grid-plus-mean color averaging. The CPU Gate1
		  reference at render16/site9/2f separates the forks: naive
		  `stratified_grid` is negative (`10.419/9.692` train/heldout PSNR versus
		  `11.862/12.671` legacy), `legacy_pixel_mean` is positive
		  (`13.025/14.614`), and `legacy_frame_pixel_mean` is the current best
		  heldout CPU candidate (`13.029/14.617`). `legacy_frame_patch3_mean` is
		  also positive versus legacy sparse (`12.761/14.315`) but worse than the
		  one-pixel frame-local candidate, so it is retained as a non-selected
		  positive. `stratified_pixel_mean` raises train PSNR to `13.679` but drops
		  heldout to `12.611`, so it is recorded as a rejected train-overfit fork
		  rather than a next-MPS candidate. CPU Gate4 compiler smokes also pass for
		  the frame-local and patch3 candidates. This is still CPU reference
		  evidence, not a clean MPS train/speed artifact; the next quiet real32 run
		  should try `legacy_frame_pixel_mean` first. The
		  site-initialization quality bridge report
		  `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_site_initialization_quality_bridge.json`
		  makes that handoff machine-readable:
		  `next_mps_candidate=legacy_frame_pixel_mean`,
		  `positive_candidate_count=3`, and `rejected_candidate_count=2`. The CPU
		  topology/capacity probe now also
		  accepts `--site-initialization`; the tiny
		  `2026-05-20_gate4_affine_candidate_csr_capacity_legacy_frame_pixel_mean_render8_site4_2_4f.json`
		  artifact passes with sublinear candidate/storage scaling for the selected
		  initializer. The combined readiness artifact
		  `research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_next_mps_candidate_readiness.json`
		  now fails closed unless the CPU quality bridge and matching topology probe
	  both pass; it records `ready_for_quiet_mps_quality_speed_run=true` while
	  keeping `quality_claim=false` and `speed_claim=false`. The fail-closed
	  launcher
		  `research_experiments/world_foam_lane2/run_worldfoam_next_mps_candidate.py`
		  plans the exact `legacy_frame_pixel_mean` real32 train/eval command and only
		  executes it after a clean strict preflight; the current preflight summary
		  `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_verified.json` is
		  `preflight_contended`, so no MPS artifact exists yet. The refreshed
		  summary now has top-level preflight status, blocker counts/reasons, and a
		  compact blocker list; the latest written blocker reasons are `high_cpu`,
		  `keyword:torch`, and `periodic_mps_exporter`, with `8` blocking rows:
		  high-CPU `font_maker`, high-CPU `ai_trader` monitor/check/export
		  children, the detached TOTO monitor chain, and a `keyword:torch` queue
		  wrapper. Earlier live samples also caught an active TOTO MPS export as
		  `keyword:mps`, so one quiet CPU snapshot is not enough. The
	  launcher now requires all requested stability samples before train/eval;
	  the current plan asks for `3` clean samples at `5s` spacing, while the
	  latest blocked artifact completed only sample `1/3` and kept
	  `preflight_stability_ok=false`. The launcher summary now also records
	  `result_verifier_command`, pointing at
	  `research_experiments/world_foam_lane2/verify_worldfoam_next_mps_candidate_result.py`.
		  That verifier is the post-run acceptance gate for this candidate: it
		  requires executed `status=train_eval_ok`, all stability samples clean, a
		  clean benchmark environment, matching `legacy_frame_pixel_mean`/native-cutwalk
		  train command and artifact, real MPS rows for every requested frame count,
	  numeric PSNR/L1, and sublinear total/backward acceptance. The current
	  blocked preflight summary fails the verifier, as intended. The refreshed
	  plan summary now sets `verify_result=true`; future clean executions should
	  pass `--verify-result` so the launcher itself returns
	  `result_verification_failed` if the post-run audit rejects the artifact.
	  The launcher now also has opt-in whole-sequence retry knobs
	  (`--preflight-retry-timeout-s`, `--preflight-retry-poll-s`): unit coverage
	  proves a dirty first preflight can be followed by a clean full stability
	  sequence before training launches, but the live retry smoke
	  `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_retrywait_smoke.json`
	  still failed closed as `preflight_contended` and produced no train/eval
	  artifact. A longer strict retry execution,
	  `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_final_try.json`,
	  made `11` preflight attempts over a `180s` retry window and still failed
	  closed as `preflight_contended`: latest blockers were high-CPU
	  `font_maker`, high-CPU `ai_trader` live quote/export/pytest children, the
	  detached TOTO periodic exporter screens, and a `keyword:torch` queue
	  wrapper. It completed only sample `1/3`, produced no `.worldfoam.json`, and
	  the verifier rejects it as expected. Current focused verification is
	  `py_compile` OK plus `33` focused tests passing; `verified_retry2`,
	  `retrywait_smoke`, and `final_try` all fail the post-run verifier for the
	  expected missing-clean-artifact reasons. A later preflight-only recheck,
	  `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_screen_blocker_recheck.json`,
	  also failed closed at sample `1/3`: high-CPU `font_maker`, high-CPU
	  `ai_trader` pytest/export children, the TOTO periodic exporter screen, and
	  a `keyword:torch` queue wrapper were still present. The latest
	  preflight-only artifact,
	  `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_actionable_blockers.json`,
	  records the same failure mode with the new
	  `preflight_external_blocker_summary`: `2` high-CPU external jobs, `1`
	  torch worker, and `5` periodic exporter processes, with no train/eval
	  artifact. A fresh
	  `2026-05-21_worldfoam_next_mps_legacy_frame_pixel_mean_current_preflight.json`
	  recheck also failed closed before training: `font_maker`, ai_trader
	  pytest/export children, the torch queue wrapper, and the TOTO exporter
	  chain were still classified as benchmark blockers. A later
	  `2026-05-21_worldfoam_next_mps_current_status_recheck.json` also failed
	  closed before training, now with high-CPU `font_maker` PID `92641`
	  (`209.2%` CPU), the `keyword:torch` queue wrapper, and the same TOTO
	  exporter chain. The source-only native
	  variant verifier
	  `2026-05-21_worldfoam_native_variant_source_wiring.json` also passes for
	  the `fused_direct`, `fused_csr`, and `fused_slab` forks, checking
	  `TORCH_LIBRARY` schemas, `m.impl` registrations and dispatch-target source
	  definitions, Python `torch.ops` wrappers, host-loaded Metal kernel names
	  against actual `kernel void` declarations in the dynamically loaded
	  `.metal` source files, and `MetalKernels` field declarations/initializers/
	  uses without requiring a Metal build. The wrapper import path now uses
	  `torch.ops.load_library` for these pure `TORCH_LIBRARY` extensions instead
	  of swallowing the expected missing-`PyInit__C` import failure. The import
	  verifier `2026-05-21_worldfoam_native_variant_import_registration.json`
	  proves normal package import registers all compiled schemas: direct
	  `11/11`, CSR `13/13`, slab `103/103`. The three forked extensions were
	  rebuilt from source with `setup.py build_ext --inplace`; regenerated import
	  metadata records fresh rebuilt `_C.cpython-311-darwin.so` mtimes. Rebuilt
	  MPS correctness smokes pass for direct/CSR/slab power-boundary counts
	  against the CPU fixture, and the slab mixed MPS regression suite passes
	  `8` tests over ownerupdate, sample-reduce, framegroup cached, and high-cap
	  replay kernels. Additional rebuilt real-ray smokes now pass for direct
	  shared real-ray replay, CSR affine moving rays, slab affine VJP without
	  ownerupdate, and slab per-track ownerupdate/VJP. The earlier slab
	  ownerupdate failure was an invalid smoke invocation: the default `tiled`
	  layout does not execute ownerupdate kernels, so the smoke now errors if
	  `--include-ownerupdate` is used without `--layout per-track`, with a new
	  focused CLI regression. The rebuilt-native smoke-bundle verifier
	  `2026-05-21_worldfoam_rebuilt_native_smoke_bundle_verifier.json` requires
	  the seven valid rebuilt smoke artifacts and classifies the old failed
	  tiled-ownerupdate artifact as `expected_invalid_tiled_ownerupdate`.
	  The goal-state report
	  `2026-05-21_worldfoam_fork_shader_goal_state.json` records
	  `shader_fork_smoke_state_fixed=true` but `objective_complete=false` and
	  `status=blocked_external_environment` because the clean real32 MPS
	  PSNR/speed/sublinear gate still has no artifact. Commit/handoff scope is
	  recorded in
	  `research_experiments/world_foam_lane2/2026-05-21_worldfoam_fork_shader_commit_scope.md`,
	  including the submodule source directories to preserve and generated
	  `.so`/`build`/`__pycache__` outputs to exclude. Current source/import/
	  rebuilt focused verification is `51` tests passing plus the `8`-test MPS
	  slab suite. A
	  later fresh preflight-only check,
	  `2026-05-20_worldfoam_next_mps_legacy_pixel_mean_fresh_preflight.json`,
	  also stopped at sample `1/3` with `8` blocking rows: high-CPU `font_maker`,
	  high-CPU `ai_trader` pytest/report children, the TOTO monitor chain, and a
	  `keyword:torch` queue wrapper. A follow-up probe
	  `2026-05-20_worldfoam_next_mps_legacy_pixel_mean_probe2351.json` also
	  failed closed at sample `1/3`, now catching a live TOTO quote snapshot and
	  multiple high-CPU pytest/RL children. The later
	  `2026-05-20_worldfoam_next_mps_legacy_pixel_mean_probe2354.json` also
	  failed closed at sample `1/3` with high-CPU `font_maker`, high-CPU
	  `ai_trader` imitation/integrity pytest children, the TOTO monitor chain,
	  and a `keyword:torch` queue wrapper. The
	  train/eval and STAR-compare benchmark captures now ignore
	  the current process ancestor chain too, so an
  `rtk sh -lc ... powerfoam_metal...` launch wrapper cannot self-block as `keyword:metal`. Native owner-run cutwalk parity now also
  covers a synthetic non-repeated `32f` moving-ray boundary for both CPU cutwalk
  delta parity and MPS framebitmask fused-shader output parity. A direct
  low-level MPS regression also forces `track_frame_mask_i32 = -(1 << 31)` and
  verifies the frame-31 shader change against an all-base tape. The MPS wrapper
  now rejects framebitmask tapes whose per-track mask popcount does not match
  the per-track change-record span. The CPU tape builder also rejects unsorted
  per-track change frames, because the shader maps frame bits to change rows by
  bit-rank/popcount and requires strictly ascending change-frame records. The
  framebitmask helper now rejects malformed change-offset vectors directly:
  empty offsets, nonzero first offsets, nonmonotonic offsets, and final offsets
  that do not match `change_frame_i32` length. The MPS wrapper has direct
  negative tests for illegal framebitmask bits as well: frame `0` and bit
  `frame_count` are rejected with popcount held constant. The same sparse-change
  validation is now shared by frame-select helper prep, which rejects unsorted
  per-track frames, frame-0 changes, and non-1D offset tensors before building
  the int16 rank map. The framebitmask MPS wrapper now also rejects empty
  `change_offsets_i32` directly instead of falling through to a generic offset
	  validator with a negative inferred change count. It also validates packed
	  base/change endpoint records in the wrapper, so invalid owner/cut codes fail
	  before reaching the Metal shader. That packed endpoint-record guard is now
	  shared by the sibling non-framebitmask wrappers too: packed recompute,
	  factorized packed, factorized frameselect, packed scalar, smallrun16,
	  materialized, and framegroup16 paths validate base/change records before
	  launching the custom op. All delta direct-config paths now require a
	  prep-time `delta_packed_records_validated` marker bound to the current
	  launch contract: raw/i16/packed record tensors, topology/config tensor
	  identities and PyTorch version counters, selector-flag presence, launch
	  scalar fields, site count, and runtime track/frame counts. Manually
	  assembled direct-config tapes and stale markers after record, topology, config, or selector
	  replacement/in-place mutation cannot bypass wrapper validation and jump
	  straight into the native Metal op. Prepared tapes set that marker after CPU
	  packed-record range validation and native config tensor creation without
	  adding per-step record copies. The prepared factorized packed,
	  frameselect, and framebitmask paths also require that current marker, so
	  stale or hand-mutated factorized prepared tapes fail before native Metal
	  launch. Handcrafted framebitmask shader fixtures now stamp the marker only
	  after their deliberate malformed mutation, preserving deeper wrapper
	  validation coverage. A selector-family regression now proves the marker is
	  required across raw, packed scalar/framegroup/materialized/recompute/
	  smallrun/launch-only, i16x4, i16cols, i16x3, and factorized selectors.
	  Additional stale-marker regressions now corrupt a launch-only scalar,
	  replace rowdesc buffers, and replace i16x3 owner-reduce chunk-owner
	  topology after the marker, and reject runtime track/frame-count mismatches
	  after the marker. Runtime tensor guards now reject malformed `site_rgba`
	  and `target_rgb_track` shape, dtype, device pairing, and contiguity before
	  native launch. The direct-config marker check also validates marked tape
	  tensor dtype, device, contiguity, tensor layout/rank, fixed ABI shapes,
	  flattened packed-record divisibility, selector compatibility, and scalar
	  contract consistency. Its marker payload is now `delta_direct_config_v8`;
	  scalar marker entries are type-stable rather than `int(...)`-coerced, so
	  invalid scalar types reach the scalar-contract validator instead of
	  failing or normalizing during marker construction. Packed i32
	  direct-config prep now stamps
	  `delta_coeff_boundary_count` plus `delta_launch_*` record/count scalars
	  for every packed selector, not only launch-only variants, and the validator
	  now requires those scalars for all i32 packed direct-config selectors. It
	  derives `delta_launch_change_count`
	  from `change_frame_*` or `change_offsets_* - 1` so factorized frameselect and
	  framebitmask tapes cannot hide stale or deleted re-stamped change counts. A
	  manually re-stamped bad tape cannot bypass the Python boundary and reach
	  Metal with malformed boundary, track-ray, rowdesc, packed-record, stale
	  launch-count, base/change record-count, missing scalar-contract, or
	  ambiguous selector payloads.
	  The packed-owner regression module now passes `55` tests after adding
	  prepared-tape corruption checks, launch-contract direct-config marker
	  guards, runtime/tape-storage guards, scalar launch-contract guards,
	  direct-config tensor-layout guards, selector-contract guards, and
	  missing scalar-contract coverage across every i32 packed selector family.
	  The latest strict real32 clean-evening wrapper attempt started from a quiet
	  preflight and did run one true-32f WorldFoam pass:
	  `2026-05-20_real32_strict_mini_wrapper_clean_evening.attempt1.worldfoam.json`
	  records `loaded_frame_count=32`, `repeat_loaded_frames=false`, `3.104ms`
	  total, `2.773ms` backward, train PSNR `12.987`, and heldout PSNR
	  `14.229`. It is still diagnostic, not promotable: the post-run benchmark
	  snapshot found restarted live `ai_trader` offline TOTO MPS-export monitors
	  plus transient `MTLCompilerService`, attempt 2 preflight stayed contended,
		  and the wrapper summary ended `worldfoam_preflight_failed_or_contended`.
		  No STAR compare command ran, so WorldFoam/STAR timing and PSNR remain
		  unpromoted until the TOTO MPS-export screens are stopped/finished or the
		  gate runs in a clean machine window. Follow-up wrapper hardening now keeps
		  `planned_star_compare_artifact` separate from selected
		  `star_compare_artifact`; failed or preflight-contended summaries leave the
		  selected STAR artifact null and only record latest attempt/written STAR
		  paths when a STAR command actually runs. The promotion verifier now also
		  requires exactly one promotable STAR attempt, with latest-attempt and
		  latest-written STAR paths matching the selected artifact.
- PowerFoam post-audit work is separate from the V-JEPA/token-GS lane unless
  the user explicitly asks to merge them.
- Hourly read-only CTO code review now lives at
  `.agents/thread_types/cto_code_reviewer_super_autist/`; it reviews recent
  commits plus staged, unstaged, and untracked disk state and writes reports
  under ignored `outputs/code_reviews/` when scheduled.

## Active Experiment Lanes

See `EXPERIMENTS.md` for configs, logs, result JSONs, W&B ids, and decisions.

Current lanes:

- STAR UVT source-view overfit and shader timing.
- STAR UVT feature-tube shader port.
- Gaussian 300-clip V-JEPA/static-dynamic scale training.
- Mixed same-view plus multicam heldout trainer bridge.
- V-JEPA/F32 multicam heldout benchmark contract.
- WorldFoam Gate4 fused-MSE/high-cap shader research.
- WorldFoam paper math appendix and cell-path proof fixture.
- Three-lane tiny visual comparison for WorldFoam/PowerFoam Metal,
  WorldTubes/STAR UVT Metal, and base dynamic 3DGS fast-mac Metal
  (`outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_report.md`);
  direct disk media is verified for all three lanes, and the clean all-lane
  summary is
  `outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_clean_all_lanes.json`.
  The same harness now has a green 128px medium tier at
  `outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_medium_128_report.md`
  and a green 128px capacity tier at
  `outputs/visual_comparisons/2026-07-07_three_lane_visual_compare_capacity_128_report.md`.
- Browser WebGPU dynamic splat training prototype at
  `web/dynaworld_browser_trainer/`. It preloads the local Neural3D
  `coffee_martini` preview when served from the repo root, falls back to a
  deterministic D-NeRF-style fixture, trains a compact dynamic splat/tube field
  in WGSL, and renders the current result live. The current app has selectable
  World Tubes-style and dynamic-splats-style approximation modes plus
  deterministic target-grid/color initialization, now defaulting to 768 splats
  after the post-95%-sampler capacity retest. The latest pass preserves the native video aspect
  ratio, uses a static mean background plus fixed-order source-over dynamic
  residual splats, uses aspect-aware Gaussian distances, and uses
  `posRadius.z` as a soft temporal gate with a visible support slider now
  defaulting to `0.30`. Training
  also now truncates Gaussian support at the same 3-sigma footprint that render
  draws. Wide side-by-side preview videos are cropped to a single source-view
  pane by default. `converge17` adds deterministic grid/motion validation
  readbacks and a motion-biased sample buffer; the first reload exposed the real
  issue as tiny `Grid Loss` but high `Motion Loss`, and short runs now lower the
  motion metric while remaining far from native-shader parity. `converge18`
  raises the default LR to `0.90` based on a short browser sweep and fixes a
  reset/ResizeObserver half-initialized render race. `converge19` adds
  motion-aware initialization by seeding later-drawn splats from the motion
  frame/pixel buffer, cutting the short-run motion loss materially. `converge20`
  changes visible `Motion Loss` to direct loss over the packed motion frame/pixel
  set instead of a grid-weighted proxy. `converge21` adds visible train
  throughput (`Steps/s`) and keeps LR defaults consistent with the UI; the latest
  Dynamic splats-style smoke showed 384 splats at `16.7` steps/s with true
  `Motion Loss 0.007612` and no new browser warnings/errors. `converge22`
  promotes 512 splats as the default because it reached true `Motion Loss
  0.006685` by step `553` at `11.6-13.5` steps/s, nearly matching 768 quality
  without the 768 throughput hit. The post-edit smoke loaded 512 by default and
  reached true `Motion Loss 0.007406` by step `174` at `12.0` steps/s with no
  new warnings/errors. `converge23` promotes temporal support `0.30` because it
  reached true `Motion Loss 0.006591` by step `574`, better than the matched
  `0.26`/`0.22`/`0.18` probes. The post-edit smoke loaded `converge23` and
  reached true `Motion Loss 0.007114` by step `237` with no new
  warnings/errors. `converge24` exposes the motion/uniform sampler split as a
  `Motion Mix` slider, and the follow-up sweep rejects LR decay to `0.45` and
  128 samples/step while promoting a 95% motion-sample mix as `converge25`:
  Dynamic splats-style reached true `Motion Loss 0.006320` by step `831`,
  versus `0.006431` by about step `810` for the old 75% mix, with no browser
  warnings/errors. `converge26` rechecks capacity under the new sampler and
  promotes 768 splats as the current default: 768 reached true `Motion Loss
  0.006201` by step `522`, while a matched 512-splat rerun reached only
  `0.006505` by step `565`, both with no browser warnings/errors. This is a
  source-view browser prototype, not a Metal parity, shared-backward, or
  benchmark row. `converge27` adds model-health diagnostics (`Motion Cov` and
  `Active` splats), showing the old path improved motion loss while dropping
  motion coverage from about `39.8%` to `29.9%`. `converge28` responds by
  seeding 48% of splats from motion samples with slightly broader/more opaque
  support; it raises initial coverage to about `63%` and reaches true
  `Motion Loss 0.005459` by step `854` while keeping coverage around `38%`,
  with no browser warnings/errors. `converge29`/`30` surfaces peak motion alpha
  plus mean opacity/radius and keeps the desktop rail scrollable, so follow-up
  reads can distinguish under-coverage from splat shrink/fade. `converge31`
  adds a small motion-sample-only coverage hinge below 50% dynamic alpha
  coverage, but the first 50%/0.20 setting over-preserves support and slows
  motion-loss improvement. `converge32` weakens it to a late 44%/0.08 guard in
  the simplified browser train shader; the extended trace reached step `861`,
  true `Motion Loss 0.005914`, and motion coverage `47.0%`, a small MSE
  tradeoff versus `converge28` but a healthier support state. `converge33`
  renames the selector to `Motion Model` and keeps World Tubes-style as the
  default because it effectively ties Dynamic splats-style under the guard:
  step `629`, true `Motion Loss 0.005938`, motion coverage `47.4%`.
  `converge34` keeps the train math unchanged and fixes the browser debugging
  surface: equal-width target/render panes and an RGB versus amplified
  motion-residual target selector. The post-edit live trace reached step `268`,
  true `Motion Loss 0.006505`, and motion coverage `50.2%` without browser
  warnings/errors. `converge38`/`39` adds result-side dynamic-layer and
  alpha-support views; this shows the moving person is covered, but broad
  background support remains active. `converge39` lowers the temporal gate
  floor (`sigma*0.70 -> sigma*0.30`), improving boot true motion loss
  `0.011522 -> 0.011099` and reaching step `72`, true
  `Motion Loss 0.007788`, motion coverage `53.9%`. `converge40` adds
  `Static Cov`, a low-motion alpha penalty, and opacity decay, but the first
  `0.055` decay weight is too blunt and drops motion coverage to `42.4%` by
  step `239`. `converge41` lowers decay to `0.025`, reaching step `294`, true
  `Motion Loss 0.006751`, motion coverage `44.6%`, `Static Cov 2.6%`, and
  `Active 406/768`; `converge42` keeps that train math while thinning the
  static-coverage validation pass to reduce readback overhead. `converge43`
  adds a dedicated low-motion sample buffer and reserves 8% of train samples for
  static cleanup, fixing the v42 issue where the low-motion penalty was mostly
  starved by the 95% motion sampler. The first v43 browser trace loaded `Static
  Px 16384` and reached step `259`, true `Motion Loss 0.006803`, motion coverage
  `45.5%`, `Static Cov 2.6%`, and `Active 420/768` with no browser
  warnings/errors. `converge44` exposes the reserve as a `Static Mix` slider:
  `0%` recovers the v42-style sampler, and the default `8%` gives effective
  `Motion Mix 92%`. The in-app smoke loaded v44 assets, stepped once, and had no
  browser warnings/errors. The matched v44 control shows the static reserve is
  not the core convergence issue: `Static Mix 0%` reached step `274`, true
  `Motion Loss 0.006794`, and motion coverage `45.0%`, while default `8%`
  reached step `271`, true `Motion Loss 0.006822`, and motion coverage
  `45.3%`. `converge45` exposes the hidden motion-support target as
  `Support Guard` and defaults it to `52%`; the first v45 trace reached step
  `297`, true `Motion Loss 0.007060`, motion coverage `48.2%`, `Static Cov
  2.7%`, and `Active 406/768`. Treat this as a support-health control, not
  renderer/init parity. `converge46` adds frame-motion centroid velocity
  initialization for motion-seeded splats and reaches step `290`, true
  `Motion Loss 0.007036`, motion coverage `47.0%`, `Static Cov 2.8%`, and
  `Active 407/768`; it is a small fit win but not a support win. `converge47`
  replaces the global velocity with a local residual-match velocity and reaches
  step `279`, true `Motion Loss 0.006885`, motion coverage `48.1%`,
  `Static Cov 2.7%`, and `Active 414/768`. `converge48` is a preview/UI pass:
  preview time loops by default and the source/target crops from the side-by-side
  Neural3D video are shown together as a camera strip. This does not change the
  training contract: the model still optimizes the 128x128x8 source-view crop,
  not a target-camera or heldout-view objective. `converge49` adds sparse-grid
  `Val MAE`, `Val PSNR`, and global-luma `Val SSIM` readouts plus a throttled
  source-view validation-error heat map. These are validation diagnostics only;
  training still uses RGB reconstruction plus the existing support/alpha
  regularizers.
  `converge50`-`59` install the paper-backed optimizer/density-control spine:
  Adam moments, persistent absolute-gradient/contribution stats, and fixed-cap
  prune/recycle/spawn into localized high-residual motion support. Readbacks are
  serialized and validation pauses training. The implementation is browser-
  green, but current quality probes are neutral/slightly negative versus
  `converge47`, so this is not a promoted convergence or heldout-view result.
- PowerFoam post-audit research backlog.

## Documentation Update Rules

When a lane changes:

1. Add a dated loose note under `agent_notes/loose_notes/`.
2. Update `EXPERIMENTS.md` if the lane has configs/logs/results.
3. Update `BASELINES.md` if the run is a benchmark or baseline row.
4. Update `TODO/README.md` if priorities or next steps changed.
5. Update `agent_notes/key_learnings.md` only if the result changes future
   reasoning, not for ordinary progress.

Do not make `AGENTS.md` a long experiment journal. Keep it as the rules and
startup router, and put experiment detail in this index plus `EXPERIMENTS.md`.
