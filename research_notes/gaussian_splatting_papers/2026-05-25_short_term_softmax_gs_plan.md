# Short-Term Plan: Finish STAR Support Gate, Park Softmax-GS Promotion

Date:
    2026-05-25
Updated:
    2026-05-28

Scope:
    Next 1-3 focused work chunks after the Softmax-GS read/implementation.
    This plan answers what to do now, not what the final representation should
    be.

## Current Plan Snapshot: 2026-05-28

What work do we have to do now?

1. Do not start with a STAR/WorldFoam Softmax-GS port. Softmax-GS is already a
   working opt-in dynamic-GS renderer probe, and its repeat evidence is mixed.
   It is not yet a representation-direction promotion.
2. Treat the STAR support-birth ladder as the active lane. The selected
   selected-patch diagnostic found a real STAR UVT binner defect, not just an
   objective weakness: chunk-shifted moving tubes with valid analytic alpha
   were dropped because `tube_bounds` rejected small valid 3x3 determinants
   and used a fallback temporal half-extent that was only the local chunk size.
3. The shader repair is now in
   `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`:
   determinant tolerance is lowered to `max(eps^2, 1e-20)`, and the fallback
   half-extent covers the shifted local domain. The focused regression is
   `tests/test_star_uvt_feature_binning.py`.
4. Repaired pre-train selected-patch diagnostics show the born support actually
   contributes: targetinit/targetalpha/targetarea2 normal patch PSNR is
   `4.606/4.686/4.684`, forced-alpha PSNR is `14.529/14.694/14.677`, and
   selected-only alpha is about `0.30` instead of `0.0`.
5. The first repaired train rerun is
   `star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetarea2_binfix_from1500_lr001_50step_media.jsonc`.
   It passes with zero overflow, max tile count `110/128`, loss
   `0.889263 -> 0.863064`, support-target-area loss
   `0.253626 -> 0.217254`, RGB-probe PSNR `24.253 -> 24.453`, and feature
   loss `0.612217 -> 0.610967`.
6. The repaired train improves the selected target patches locally:
   normal/forced/oracle patch PSNR is `6.644/19.452/26.994`, patch alpha mean
   is `0.481`, and selected-only alpha is `0.444`. The next gate is therefore
   not "do the selected tubes render?" but "does this local support improvement
   transfer to dense RGB/support media and heldout/world behavior?"
7. Dense support transfer is now measured. The binfix improves whole-frame
   normal PSNR from the pre-binfix targetarea2 repair row (`6.507`) to
   `7.269`, forced-alpha PSNR from `14.085` to `14.736`, and alpha `>0.1` from
   `65.7%` to `75.4%`. But the target-background oracle remains high
   (`21.439`) and a raw opacity-bias rerender only reaches `8.039`, so the
   remaining bottleneck is still visibility/composition, not just local support
   existence or opacity scale.
8. The visibility-prefix/compositing-tape gate is now measured for 256 selected
   support-target rays. Selected born tubes are absent on `0.0%` of sampled
   rays, prefix-hidden on only `1.6%`, carry `93.1%` weight share, and are the
   top contributor on `95.7%`; local normal/forced/oracle PSNR is
   `6.522/19.129/26.831`. This ruled out "hidden selected tubes" as the main
   debug target.
9. The prefix-alpha compositing train is now measured at the same 50-step budget
   as the binfix row. It passes fixed-bin and learns the local control surface:
   total loss `1.285825 -> 1.210325`, support-target-area loss
   `0.253626 -> 0.219786`, prefix-alpha loss `0.198281 -> 0.172906`,
   selected weight mean `0.4114 -> 0.4419`, and final alpha mean
   `0.4456 -> 0.4751`. But dense support is basically flat against the
   no-prefix binfix row: `7.262/14.732/21.438` normal/forced/oracle PSNR,
   alpha `>0.1` `75.4%`, best raw-opacity-bias PSNR `8.037`.
10. The next short-term STAR train should broaden support ownership/coverage or
   change the target sampling distribution. Do not run another alpha-pressure
   repeat unless it changes which points/tubes receive ownership or adds a new
   dense/visibility signal.
11. Keep WorldFoam out of this short-term loop unless the STAR diagnostic shows
   the failure is fundamentally "fuzzy primitive ownership" rather than missing
   support/prefix behavior. Foam needs a matched tournament row, not a vibes
   promotion.

Short-term success criteria:

```text
next STAR gate improves dense normal RGB, not only local alpha/prefix loss
fixed-bin cap stays valid or the overflow is explicitly budgeted
forced-alpha/oracle gap narrows, not just shifts
```

Short-term stop criteria:

```text
another local pointwise, 2x2 patch, or prefix-alpha loss learns but dense support stays flat
another support-selection scorer improves scalar loss but not forced/oracle gap
Softmax-GS repeats only source/train wins without heldout repeat
WorldFoam has no matched heldout-quality row yet
```

## Immediate Answer: Work To Do Now

The next useful work is STAR support, not another Softmax-GS port.

Softmax-GS has already reached the intended short-term decision point:
the renderer route exists, selected scalar tape exists, K=16 is the first
usable bounded setting, and the matched dynamic-GS heldout evidence is mixed.
The tiny 64px/4f/128-splat row was positive on heldout PSNR, but the 512-splat
repeat lost heldout PSNR and the practical 128px/16f/512 stride16 row only
nudged PSNR (`12.1234 -> 12.2092`) while losing SSIM/train-view metrics.

Therefore the immediate queue is now:

1. Treat the STAR target-grid/sparse-forward ladder as regenerated through the
   selected 1500-step checkpoint.
2. Treat the `K=8/r64/o0.4` 50-step support row as measured, not pending:
   it improves loss/probe metrics but fails the hard cap-128 renderer budget
   with `tile_overflow_sum=277` and max tile count `146/128`.
3. Treat the cap-pressure reduction follow-up as measured too: `n16/r48` and
   `n16/r40` reduce overflow to only two tiles but still fail fixed-bin
   eligibility, while `n8/r40/o0.4` passes cap-128 with zero overflow and max
   tile count `123/128`.
4. Use `K=8/n8/r40/o0.4` as the current cap-128-safe STAR support seed, not as
   a visual-quality promotion. It improves loss/probe metrics and dense support
   diagnostics over the regenerated 1500 checkpoint, but dense RGB is still
   weak and forced-alpha/oracle gaps remain large.
5. Treat the longer cap-safe probe as measured: the selected 90-step checkpoint
   passes and improves loss/probe/feature loss with zero overflow, while the
   100-step endpoint fails by a late objective jump despite staying fixed-bin.
   The checkpoint-aware tail schedule is now measured too: it keeps the
   100-step row passing, but does not beat the 90-step checkpoint or move dense
   support. The next STAR move is a smarter visibility/support bridge, not
   another broad radius/opacity/LR-tail sweep.
6. Treat the uniform/all-centers follow-up as measured: uniform reallocation
   works mechanically, but `n16` still overflows by two tiles, `K=16/n16`
   still overflows by the same two tiles even with one tube per center, and
   shrinking to `r32` does not clear it. `K=12/n12/r40/o0.4` is cap-safe but
   does not beat the selected `K=8/n8` 90-step checkpoint on objective/feature
   loss, and dense support is effectively flat.
7. Treat the first cap-aware support bridge as measured: cap-slack target
   scoring picks low-load pixels but still overflows by two tiles; exact-fit
   repair clears birth-time overflow but drifts to one final overflow tile;
   guarded repair (`K=16/n16/r40/o0.4`, guard `2`) passes with zero final
   overflow and max `127/128`, but dense support is only a tiny nudge over
   `K=12/n12`. This is a useful bridge primitive, not a selected-checkpoint win.
8. Treat the first residual-cap-slack scorer as measured too: it selects
   high-residual/low-alpha points, passes fixed-bin, and improves scalar
   objective/probe slightly, but dense normal/forced/oracle support remains flat.
   Pointwise residual scoring alone is not enough.
9. Treat footprint-aware residual targeting as measured too: mean-pooling the
   residual/uncovered score over the projected support footprint gives the best
   K16 scalar endpoint so far, but dense support remains flat
   (`6.481/14.021/21.576` normal/forced/oracle). Target selection is no longer
   the main lever; the next gate needs a model handoff or alpha/composition
   change.
10. Treat target-grid feature init as measured too: it is a small positive
    model-handoff row (`0.752454 -> 0.748504`, dense
    `6.488/14.054/21.629`), but alpha coverage stays flat (`>0.1` `0.655`).
    This says feature handoff helps content/oracle a bit; the next gate should
    change alpha/composition or visibility-prefix behavior.
11. Treat support-target alpha as measured too: the pointwise sparse alpha
    objective on selected birth targets learns cleanly (`0.492962 -> 0.478448`)
    and nudges dense support (`6.508/14.084/21.626`, alpha `>0.1` `0.657`),
    but it does not materially close the forced-alpha/oracle gap. This says
    "more alpha" only matters if it is coupled to target-area composition,
    visibility-prefix structure, or a compositing tape.
12. Treat support-target-area patches as measured too: 2x2 patch-mean
    composition around selected birth targets is a cheaper local positive
    (`0.597970 -> 0.581641`, mean support-target-area time `209ms`) but lands
    on the same dense plateau (`6.507/14.085/21.627`, alpha `>0.1` `0.657`) and
    weakens feature loss versus target-init. The binner repair then moves dense
    support to `7.269/14.736/21.439`, with alpha `>0.1` `0.754`, but still
    leaves a huge forced-alpha/oracle gap. This rules out small local patch
    pressure as the next main lever.
13. Treat prefix-alpha compositing pressure as measured too: the 50-step row
    learns its local loss (`0.198281 -> 0.172906`) and raises selected
    prefix-weight contribution (`0.4114 -> 0.4419`) plus final alpha
    (`0.4456 -> 0.4751`), but dense support remains at the binfix plateau
    (`7.262/14.732/21.438`, alpha `>0.1` `0.754`). This says the next lever is
    broader ownership/coverage or a new dense visibility signal, not the same
    local prefix-alpha objective by itself.

Softmax-GS stays as an opt-in dynamic-GS renderer probe. Do not port it to
STAR or WorldFoam from the current evidence.

## Decision Question

Does Softmax-GS earn its cost as a renderer/compositing variant for the current
DynaWorld splat lanes?

The answer must be measured against:

- matched primitive count;
- matched data/config budget;
- heldout-view metrics when available;
- source-view media only as a mechanics/debug signal;
- wall-clock and backward cost.

## What We Do Now

Current priority:
    The selected STAR checkpoint ladder and `K=8/r64/o0.4` 50-step
    support-changing continuation have now been run, and the narrower
    cap-pressure follow-up has found one valid row: `K=8/n8/r40/o0.4`. The
    longer gate selects the 90-step endpoint, not the 100-step overrun, and
    the measured LR-tail schedule is a stability row rather than a better
    checkpoint.
    Softmax-GS bounded-tape behavior is characterized enough for the present
    decision: K=8 is too lossy, K=16 is viable, K=32 does not improve the tiny
    endpoint, and repeat/scale quality evidence is not a promotion. The active
    question is now how to make the support bridge visibility-aware. The binner
    repair gave a real dense support gain, but the remaining forced-alpha/
    oracle gap is still too large for another local pressure variant.

The current order/status from here is:

1. Record the regenerated 1300->1500 sparse-forward lineage, the failed
   high-pressure support row, and the cap-safe `n8/r40` row in
   `EXPERIMENTS.md`, `TODO/README.md`, and loose notes.
2. Treat `K=8/n8/r40/o0.4` from the 1500 checkpoint as the cap-128 support
   baseline: `pass=true`, zero overflow, max tile `123/128`.
3. Treat the 90-step checkpoint-selection row as the current best cap-safe
   support checkpoint: `pass=true`, zero overflow, max tile `122/128`, loss
   `0.754568 -> 0.747006`, feature loss `0.608402 -> 0.606764`, RGB-probe
   PSNR `24.372 -> 24.552`.
4. Treat the constant-LR 100-step continuation as a selection warning: it
   remains fixed-bin (`0` overflow, max tile `122/128`) but fails the final
   gate after late jumps at global steps `1590` and `1594`.
5. Treat the `lr=0.001` until global `1588`, then `lr=0.00025` tail schedule
   as a stability probe: it passes at 100 steps with zero overflow and final
   loss `0.749454`, but dense support is effectively the same as the 90-step
   row (`6.462` normal PSNR, `14.012` forced-alpha, `21.578` oracle).
6. Record the uniform-allocation and `K=12/n12` follow-up as a negative/neutral
   bridge attempt. `n16` uniform, `K=16/n16`, and `K=16/n16/r32` all miss
   cap-128 by two tiles; `K=12/n12/r40` passes but its 90-step row ends worse
   than the selected `K=8/n8` 90-step checkpoint (`0.749217` vs `0.747006`
   loss; `0.608311` vs `0.606764` feature loss), while dense normal/forced/
   oracle PSNR stays around `6.47/14.01/21.55-21.58`.
7. Build a stronger alpha/composition bridge; the measured rows now show
   that proportional clustering, uniform allocation, center count, radius
   shrinkage, LR-tail scheduling, tile slack, and pointwise residual scoring are
   not enough. Footprint-aware residual scoring improves the scalar endpoint but
   not dense support. Target-grid feature init improves content/oracle slightly
   but leaves alpha coverage flat. The first support-target alpha objective
   learns its local loss, but only moves dense support from
   `6.488/14.054/21.629` to `6.508/14.084/21.626`. Guarded tile repair is
   available as a cap-safe primitive. The support-target-area 2x2 bridge is
   cheaper than pointwise alpha but lands at `6.507/14.085/21.627`. The binner
   repair moves dense support to `7.269/14.736/21.439` with alpha `>0.1`
   `75.4%`, but the forced-alpha/oracle gap remains large and raw opacity bias
   reaches only `8.039` PSNR. Prefix tape then shows selected support is present
   and dominant locally. The prefix-alpha compositing objective learns locally
   but does not move dense support beyond the binfix plateau, so the next bridge
   should broaden ownership/coverage or change the support sampling distribution.
8. Only if STAR support has adequate coverage and remaining artifacts are
   visibly overlap/order-related should Softmax-GS get a STAR CPU diagnostic.

The most important stop rule:
    Do not spend STAR or WorldFoam engineering time on Softmax-GS until either
    dynamic GS proves a repeatable heldout/rate win or STAR proves its remaining
    bottleneck is overlap/order after support coverage is no longer the main
    failure.

## Current Belief

Dynamic GS is the right first target. It is close enough to classic 3DGS that
Softmax-GS tests the intended failure mode: sparse/fuzzy splat overlap,
boundary sharpness, and camera-motion popping.

Softmax-GS should get only a STAR CPU/reference diagnostic until the dynamic-GS
probe is positive or support coverage improves. STAR's current blocker is
missing target support and target-area composition, not only bad blending of
existing support. A pointwise support-target alpha loss now learns but barely
moves dense support; a 2x2 support-target-area patch loss also learns but lands
on the same plateau. The binner repair is a partial dense support gain, not a
closeout. The next alpha/composition work needs prefix/order context, not just
more local pressure, and the first prefix-alpha train says that local prefix
pressure alone is still insufficient.

WorldFoam should not receive Softmax-GS work in the short term. Foam has a
different geometry contract; Softmax-GS mostly patches Gaussian overlap.

## Immediate Work Queue

Do these in order:

1. Softmax-GS Torch reference/tests, depth plumbing, RGB renderer fork,
   no-op parity, Metal forward, native backward, bounded tape, selected scalar
   VJP, tiny trainer smokes, heldout rows, and residual diagnostics are done
   for the decision we needed.
2. Regenerate the missing STAR support dependencies. Done for the
   RGB-probe/colorizer, V-JEPA feature cache, and checkpoint ladder through
   the selected 1500-step sparse-forward checkpoint.
3. Run the selected 50-step birth/split continuation from 1500. Done; it is a
   useful negative gate because quality/probe metrics improve but tile overflow
   fails (`277` overflowed tiles, max tile `146/128`).
4. Run a cap-128-safe support follow-up before any larger architecture move.
   Done: `n16/r48` and `n16/r40` still overflow by two tiles; `n8/r40` passes
   with zero overflow and keeps the objective/probe moving.
5. Run the longer `n8/r40` gate. Done: 90-step checkpoint-selection passes,
   100-step overrun fails by late objective jumps while remaining fixed-bin.
6. Treat the 2026-05-26 binner repair and targetarea2 binfix rerun as measured:
   selected support now renders locally and dense support improves, but the
   whole-frame result remains visibility/composition limited.
7. Treat the compact visibility-prefix tape diagnostic as measured too:
   selected support owns the sampled target rays and is not hidden. The
   prefix-weight/final-alpha train is now measured too: it learns the local
   contribution objective but leaves dense support essentially flat. Next, use
   broader ownership/coverage sampling before any Softmax-GS STAR port,
   WorldFoam switch, or another local support-target loss.

The first reference/depth task, bounded-tape shader lowering, tape-backed
color-gradient consumer, selected scalar VJP path, first tiny matched heldout
diagnostic, primitive-count repeat, tape residual diagnostic, and a practical
128px/16f stride16 repeat are now done. The next real task is STAR
visibility-prefix/composition, not Softmax-GS integration into STAR/WorldFoam.

## Status Update: 2026-05-26

Current routing:

- Softmax-GS is parked as an opt-in dynamic-GS probe unless a learned-parameter
  row or stronger heldout split reverses the mixed repeat evidence.
- STAR support is the active short-term lane. The old 1500-step
  sparse-forward checkpoint lineage has been regenerated locally, the preflight
  passes cleanly, and the 50-step `K=8/r64/o0.4` birth/split continuation has
  been run. It fails the final gate only because cap-128 tile overflow becomes
  nonzero (`277` overflowed tiles, max tile `146/128`), while total loss,
  feature loss, and RGB-probe loss all improve. The narrower follow-up is now
  measured: `n16/r48` and `n16/r40` reduce overflow to two tiles but still fail,
  while `n8/r40` passes (`0` overflow, max tile `123/128`) and improves loss
  `0.754568 -> 0.749460`, feature loss `0.608402 -> 0.607554`, RGB-probe PSNR
  `24.372 -> 24.501`, and dense support versus the regenerated 1500
  checkpoint. The longer safe-row probe then selects the 90-step checkpoint
  (`pass=true`, zero overflow, max tile `122/128`, loss
  `0.754568 -> 0.747006`, feature loss `0.608402 -> 0.606764`, RGB-probe PSNR
  `24.372 -> 24.552`) and rejects the 100-step endpoint, which stays fixed-bin
  but regresses loss after late jumps. The checkpoint-aware 100-step schedule
  (`lr=0.001` through global `1587`, then `0.00025`) passes and avoids the
  catastrophic constant-LR endpoint (`0.749454` final loss, zero overflow), but
  still loses to the selected 90-step checkpoint and leaves dense support flat:
  normal/forced/oracle PSNR `6.462/14.012/21.578`. The next allocation probe is
  measured too: uniform `n16`, `K=16/n16`, and `K=16/n16/r32` still fail
  cap-128 by two tiles (`131/128` max), so the last saturated tiles are not
  just a proportional-packing artifact. `K=12/n12/r40/o0.4` is cap-safe
  (`127/128` at 50 steps, `126/128` at 90 steps) and has the best 50-step dense
  normal PSNR in this small table (`6.483`), but the 90-step row loses the
  selected-objective comparison and does not move forced-alpha/oracle support
  (`14.013/21.551`). Keep `K=8/n8/r40/o0.4` 90-step as the selected checkpoint.
- WorldFoam remains a challenger lane, not the immediate answer to the
  Softmax-GS finding.

Completed evidence:

- `research_experiments/softmax_gs/reference.py` and
  `tests/test_softmax_gs_reference.py` cover vanilla parity, same-depth
  two-splat order invariance, separated-depth fallback, transmittance
  preservation, finite gradients, contribution-tape reconstruction, and exact
  color-gradient weights. They now also cover bounded top-K tape selection in
  ray order plus a residual-weight output-error bound for unit-range features.
- `src/train/renderers/fast_mac.py` now has `depth_mode` with
  `rank_depth` default and `center_camera_z` for Softmax-GS experiments.
- `v5_softmax_gs` is a config-selected fast-mac RGB variant.
- No-op `v5_softmax_gs` matches vanilla `v5` on MPS:

  ```text
  forward_max_abs = 0.0
  grad_max_abs = 2.98e-8
  ```

- Softmax-GS compositing is implemented in the fast eval/state and overflow
  Metal forward paths. A same-depth red/blue two-splat MPS test shows the
  intended effect:

  ```text
  vanilla swapped-order max diff = 4.7309e-1
  softmax swapped-order max diff = 2.3842e-7
  ```

- The contribution-tape reference gate passes:

  ```text
  PYTHONPATH=src/train uv run --with pytest python -m pytest \
    tests/test_softmax_gs_reference.py -q

  11 passed
  ```

- The native fast-tile plus overflow recompute/tape backward gate passes:

  ```text
  PYTHONPATH=src/train uv run --with pytest python -m pytest \
    tests/test_fast_mac_depth_signal.py \
    tests/test_softmax_gs_reference.py \
    tests/test_softmax_gs_metal_forward.py \
    tests/test_fast_mac_feature_background.py -q

  28 passed
  ```

- The bounded top-K tape now has a Metal ABI/kernel lowering for fast and
  overflow tiles. `rasterize_softmax_gs_bounded_tape(...)` returns selected
  IDs, selected weights, residual mass, and final alpha. The fast and
  forced-overflow MPS tests match the Torch bounded-tape reference exactly
  within float tolerance.
- The bounded tape now has backward consumers for color and selected scalar
  geometry/opacity/depth gradients. With a full tape
  (`softmax_gs_tape_k >= active contributors`), fast and forced-overflow
  gradient tests match the Torch reference for means, conics, colors,
  opacities, and depths. With bounded K, the path follows the same
  selected-contributor approximation contract as the forward tape.

- Native enabled trainer smokes pass with W&B disabled/offline as mechanical
  shader-route checks:

  ```text
  one-step enabled: initial 0.4373, step 1 0.4270, 8.59s/it including compile
  five-step enabled: initial 0.4370, final 0.4445, tqdm mean 2.10it/s
  same-session no-op: initial 0.4381, final 0.4324, tqdm mean 1.62it/s
  forced-overflow enabled: initial 0.4374, final 0.4486, max_fast_pairs=1
  post-tape-ABI five-step enabled: initial 0.4382, final 0.4165
  post-tape-ABI forced-overflow: initial 0.4382, final 0.4529
  tape-scalar five-step enabled: initial 0.4382, final 0.4190, softmax_gs_tape_k=8
  ```

- A matched 64px/4f/128-splat 10-step offline W&B diagnostic now exists with
  media:

  ```text
  no-op control:
      initial 0.4330
      final 0.4177
      tqdm mean 2.80s/it
      offline run wandb/offline-run-20260525_193712-1lra1t7t

  enabled native recompute:
      initial 0.4339
      final 0.4413
      tqdm mean 1.27s/it
      offline run wandb/offline-run-20260525_193830-fu0df3ks
  ```

  This is a train/media diagnostic, not a quality promotion. W&B was run
  offline because `WANDB_API_KEY` was unset locally.

- A fresh post-overflow-shader matched 64px/4f/128-splat 10-step offline W&B
  diagnostic also exists:

  ```text
  no-op control:
      initial 0.4337
      final 0.4456
      tqdm mean 2.43s/it
      offline run wandb/offline-run-20260525_195019-27rj83gw

  enabled native recompute:
      initial 0.4342
      final 0.4198
      tqdm mean 1.60s/it
      offline run wandb/offline-run-20260525_195115-tn9t3nby
  ```

  This run is still tiny/source-view-only. It shows enabled Softmax-GS can win
  this small draw after the overflow shader work, but it is not a promotion.

- TokenGS now has a normalized `train.seed` default (`17`) and calls
  `torch.manual_seed(...)` at trainer startup. The Softmax-GS configs pin
  `seed=17`, so matched diagnostics share model initialization and temporal
  sampling. A seeded 50-step 64px/4f/128-splat offline W&B diagnostic now
  exists:

  ```text
  no-op control:
      initial 0.4338
      final 0.1467
      tqdm mean 1.65it/s
      offline run wandb/offline-run-20260525_200015-s04n74di

  enabled native recompute:
      initial 0.4338
      final 0.1512
      tqdm mean 1.32it/s
      offline run wandb/offline-run-20260525_200101-xd4sm546
  ```

  This cleaner run is neutral to slightly negative for enabled Softmax-GS on
  the tiny source-view setup. It argues against touching STAR/WorldFoam with
  Softmax-GS right now.

- Historical Torch-fallback smokes are superseded by the native route. They
  remain chronology only and should not be cited as current performance or
  quality evidence.

Post-selected-scalar diagnostics:
    The selected scalar tape path is exact under full-tape coverage and now
    trains. The first K=8 64px/4f/128-splat 50-step source-view diagnostic is
    a negative result: final loss `0.2026`, offline run
    `wandb/offline-run-20260525_204628-sk2fc3ne`. Raising runtime tape cap and
    config K to 16 recovers the earlier seeded no-op/recompute bracket:
    initial `0.4338`, final `0.1472`, tqdm mean `3.19it/s`, offline run
    `wandb/offline-run-20260525_204816-oip27eka`. Raising K again to 32 does
    not improve the endpoint: initial `0.4338`, final `0.1588`, tqdm mean
    `3.63it/s`, offline run `wandb/offline-run-20260525_205435-wy8r4v9l`.

Matched multicam heldout diagnostic:
    A first heldout-style row now exists using RGB-pyramid features to avoid
    V-JEPA cache/download cost:

    ```text
    no-op control:
        config local_mac_multicam_softmax_gs_noop_rgb_pyramid_64_4f_128splats_20step.jsonc
        initial/final train loss 0.5910 -> 0.2261
        train PSNR/SSIM view0 13.4197/0.1148
        train PSNR/SSIM view1 14.3734/0.1679
        heldout camera_0040 PSNR/SSIM 4.7369/0.0503
        step-20 total/backward/raster 291/86/58ms
        offline run wandb/offline-run-20260525_210925-39a0kpp2

    enabled K=16 selected scalar tape:
        config local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_64_4f_128splats_20step.jsonc
        command requires GSP_TAPE_CAP=16
        initial/final train loss 0.5910 -> 0.2262
        train PSNR/SSIM view0 13.4502/0.0944
        train PSNR/SSIM view1 12.3880/0.1191
        heldout camera_0040 PSNR/SSIM 11.7255/0.0794
        step-20 total/backward/raster 372/97/48ms
        offline run wandb/offline-run-20260525_211008-vfwslw6q
    ```

    This is the first result that is stronger than source-view-only evidence:
    heldout PSNR improves a lot while final train loss is tied. It is still a
    tiny 4-frame/64px/128-splat RGB-pyramid diagnostic, so it does not justify
    a baseline row or STAR/WorldFoam port yet.

Primitive-count repeat/scale diagnostic:
    The first runnable repeat raised primitive count to 512 splats while
    keeping the same 64px/4f precomputed-model shape. The original wider scale
    attempts exposed an MPS `nn.MultiheadAttention` large-memory bug before
    rasterization; a manual batch-first cross-attention fallback now fixes that
    blocker for MPS memory above 32,768 tokens. The unstrided 128px/16f route
    is technically unlocked but locally impractical, so the practical scale
    row uses `video_feature_token_stride=16`:

    ```text
    MPS forward blocker/fix:
        raw MPS MultiheadAttention crashes at 40,960 cross-attention memory tokens
        64px/4f RGB pyramid has 20,480 tokens and was safe
        64px/8f, 128px/4f, and 128px/16f exceeded the threshold and crashed
        manual MPS fallback now completes a 128px/16f forward/tape smoke
        full-memory 128px/16f training was interrupted after 3/20 steps at 9:47
        stride16 reduces 128px/16f RGB-pyramid memory back to 20,480 tokens

    runnable no-op control:
        config local_mac_multicam_softmax_gs_noop_rgb_pyramid_64_4f_512splats_20step.jsonc
        initial/final train loss 0.5817 -> 0.2511
        train PSNR/SSIM view0 11.8441/0.1112
        train PSNR/SSIM view1 12.0649/0.1218
        heldout camera_0040 PSNR/SSIM 12.5002/0.0817
        step-20 total/backward/raster 707/155/97ms
        offline run wandb/offline-run-20260525_212845-8rj3swm6

    runnable enabled K=16 selected scalar tape:
        config local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_64_4f_512splats_20step.jsonc
        command requires GSP_TAPE_CAP=16
        initial/final train loss 0.5818 -> 0.2378
        train PSNR/SSIM view0 12.8191/0.0917
        train PSNR/SSIM view1 12.0651/0.1221
        heldout camera_0040 PSNR/SSIM 11.8847/0.0950
        step-20 total/backward/raster 554/140/102ms
        offline run wandb/offline-run-20260525_212923-wbr8y46t
    ```

    This weakens the tiny-row story: K=16 improves final train loss at 512
    splats and slightly improves heldout SSIM, but it loses heldout PSNR to
    no-op by `0.6155dB`. The 128-splat heldout PSNR jump is not repeated yet.

128px/16f stride16 scale diagnostic:
    Added a practical pair that keeps 128px render, 16 frames, and 512 splats,
    but sets `model.video_feature_token_stride=16`:

    ```text
    no-op control:
        config local_mac_multicam_softmax_gs_noop_rgb_pyramid_128_16f_512splats_stride16_20step.jsonc
        initial/final train loss 0.5843 -> 0.2577
        train PSNR/SSIM view0 10.9996/0.1416
        train PSNR/SSIM view1 12.2710/0.1729
        heldout camera_0040 PSNR/SSIM 12.1234/0.1244
        step-20 total/backward/raster 1865/336/122ms
        offline run wandb/offline-run-20260525_220100-zod704i9

    enabled K=16 selected scalar tape:
        config local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_128_16f_512splats_stride16_20step.jsonc
        command requires GSP_TAPE_CAP=16
        initial/final train loss 0.5843 -> 0.2504
        train PSNR/SSIM view0 10.8973/0.1372
        train PSNR/SSIM view1 11.6462/0.1581
        heldout camera_0040 PSNR/SSIM 12.2092/0.1088
        step-20 total/backward/raster 1107/197/65ms
        offline run wandb/offline-run-20260525_220309-pkrvtzda
    ```

    This is mixed evidence. Enabled K=16 improves final train loss slightly and
    heldout PSNR by only `0.0858dB`, but loses both train-view metrics and
    heldout SSIM. It is not a clean repeat of the tiny 128-splat heldout win.

Tape residual diagnostic:
    Added a focused bounded-tape coverage script:

    ```text
    research_experiments/softmax_gs/diagnose_tape_coverage.py
    outputs/benchmarks/2026-05-25_softmax_gs_tape_coverage_64_4f_512_k16/
    ```

    It retrains the enabled 64px/4f/512 config for 20 local diagnostic steps
    with W&B disabled, decodes the initial clip, renders bounded-tape coverage
    for train0/train1/heldout0, and sweeps K=1/2/4/8/16. Summary:

    ```text
    K=16 residual/alpha mean/p99:
        train0 camera_0001 0.000652 / 0.008290
        train1 camera_0015 0.000879 / 0.009899
        heldout camera_0040 0.001930 / 0.012332

    K=8 residual/alpha mean/p99:
        train0 camera_0001 0.006965 / 0.054060
        train1 camera_0015 0.010092 / 0.057736
        heldout camera_0040 0.040167 / 0.112505
    ```

    So K=16 is not an obviously lossy tape on this row. The heldout PSNR miss
    at 512 splats is unlikely to be explained only by omitted color mass from
    bounded tape truncation.

Current blocker:
    Softmax-GS `enabled=true` now has native recompute backward and a selected
    bounded-tape backward for fast/overflow coverage. The remaining blocker is
    not "no scalar tape"; it is stronger heldout evidence. K=8 is too lossy.
    K=16 is viable and the tape tail is small on the 512-splat row, but it is
    not yet a heldout-repeat win at larger primitive count. K=32 is a negative
    source-view follow-up. The old 128px/8f MPS model-forward crash is fixed by
    the manual cross-attention fallback, but the unstrided full-memory path is
    too slow for local 20-step comparison. The stride16 128px/16f comparison is
    practical and mixed/weak. None of this promotes Softmax-GS yet.

Next implementation step:
    Do not port to STAR/WorldFoam from the current evidence. The most useful
    Softmax-only fork is learned per-Gaussian or per-layer Softmax-GS
    parameters on dynamic GS; otherwise move back to STAR support and
    WorldFoam challenger gates. The next promotion gate must show a repeated
    heldout PSNR/SSIM win, not only source/train loss, low tape residual, or
    a tiny heldout-PSNR nudge.

Concrete fallback action taken:
    STAR support is now the immediate non-Softmax path. The selected
    `K=8/r64/o0.4` birth/split row has been run for 50 steps from the
    regenerated sparse-forward 1500 checkpoint. It is not artifact-blocked
    anymore. The run is `pass=false` because the final hard overflow assertion
    trips: `tile_overflow_sum=277`, max tile count `146/128`, overflow excess
    refs `1233`. It still improves weighted loss `0.773832 -> 0.760400`,
    feature loss `0.612675 -> 0.611403`, RGB-probe loss
    `0.004029 -> 0.003725`, and RGB-probe PSNR `23.948 -> 24.289`.
    The cap-pressure reduction follow-up then found the current safe row:
    `K=8/n8/r40/o0.4` passes with zero overflow, max tile `123/128`, loss
    `0.754568 -> 0.749460`, feature loss `0.608402 -> 0.607554`, RGB-probe
    loss `0.003654 -> 0.003548`, and RGB-probe PSNR `24.372 -> 24.501`.
    Dense support improves over `start1500` (`6.035 -> 6.472` normal PSNR,
    `10.702 -> 14.018` forced-alpha PSNR, `16.787 -> 21.602` oracle), but the
    normal/forced/oracle gap says coverage/composition is still the bottleneck.
    The longer safe-row gate improves objective/probe until global step `1589`:
    the selected 90-step checkpoint passes with loss `0.754568 -> 0.747006`,
    feature loss `0.608402 -> 0.606764`, RGB-probe PSNR `24.372 -> 24.552`,
    zero overflow, and max tile `122/128`. The 100-step endpoint remains
    fixed-bin but fails by a late objective jump (`0.755682` final loss). Dense
    support is nearly flat across 50/90/100 (`6.472/6.462/6.450` normal PSNR).
    The checkpoint-aware 100-step tail schedule passes (`0.749454` final loss,
    zero overflow) and suppresses the worst constant-LR jump, but it matches the
    90-step dense support row rather than improving it. The next STAR action is
    a smarter support bridge, not more Softmax-GS engineering or schedule-only
    cleanup.

## Work Package A: Paper-To-Code Reference

Goal:
    A tiny Torch reference that proves we understand the compositing law before
    touching Metal.

Files likely involved:
    `research_experiments/softmax_gs/`
    `tests/test_softmax_gs_reference.py`

Implementation:

1. Add a CPU/Torch function for one pixel/ray:

   ```text
   inputs:
       sorted alpha [K]
       exponent p [K]
       depth d [K]
       color/feature [K,F]
       beta/gamma [K] or scalar

   outputs:
       feature/RGB [F]
       final alpha
       optional tape/debug rows
   ```

2. Keep a vanilla fallback in the same helper:

   ```text
   alpha_shape = gaussian
   beta = off
   gamma = off
   ```

3. Add tests for:

   - vanilla parity when competition is disabled;
   - two same-depth contributors swapped in input order;
   - separated-depth contributors converge back to vanilla;
   - transmittance preservation for the two-contributor case;
   - finite gradients for `alpha/beta/gamma`.

Failure it catches:
    Misreading the paper as a postprocess or a pure color softmax. If the
    reference cannot preserve alpha/transmittance on tiny cases, Metal work is
    wasted.

Exit criterion:
    Tiny reference tests pass and a Markdown report records the exact cases.

## Work Package B: Dynamic-GS Plumbing Audit

Goal:
    Make sure a shader fork would receive the right inputs.

Files likely involved:
    `src/train/renderers/fast_mac.py`
    `src/train/renderers/common.py`
    `src/train/pipeline/render.py`

Known blocker:
    `project_for_fast_mac(...)` currently passes artificial rank-depths into
    the fast-mac rasterizer. Softmax-GS needs real projected depth or
    pixel-affine depth for `gamma`.

Tasks:

1. Add a projection-side diagnostic that reports whether renderer depth is:

   ```text
   rank_depth
   center_camera_z
   pixel_affine_depth
   unknown
   ```

2. Carry true center-camera depth into the fast-mac projected interface for a
   reference path. Do not change default behavior until parity tests exist.

3. Decide the first `p` signal:

   ```text
   p = -0.5 * Mahalanobis2D
   a = opacity * exp(p)
   ```

   For dynamic GS this is the closest paper analog.

Exit criterion:
    A no-op dynamic-GS render path can consume meaningful depth and match
    vanilla output.

## Work Package C: Dynamic-GS Softmax Variant

Goal:
    One config-selected renderer variant, not a broad renderer refactor.

Files likely involved:
    `third_party/fast-mac-gsplat/variants/<new_variant>/`
    `src/train/renderers/fast_mac.py`
    `src/train_configs/...softmax_gs...jsonc`

Tasks:

1. Fork the smallest current RGB/F3 fast-mac variant first.
2. Add optional parameters:

   ```text
   smgs_alpha_shape
   smgs_beta
   smgs_gamma
   ```

   Start global/scalar before per-primitive learning.

3. Add fixed-parameter no-op mode:

   ```text
   render_mode = vanilla_equivalent
   ```

4. Add one learned-parameter version only after no-op parity passes.

5. Keep `feature_dim=3` first. F32 feature splats add colorizer ambiguity and
   should come after RGB proof.

Exit criterion:
    Vanilla-equivalent parity and one trainable dynamic-GS smoke complete.

## Work Package D: Native/Tape Backward

Goal:
    Make `softmax_gs_enabled=true` train without the slow PyTorch recompute
    scaffold.

Files likely involved:
    `research_experiments/softmax_gs/reference.py`
    `tests/test_softmax_gs_reference.py`
    `third_party/fast-mac-gsplat/variants/v5_softmax_gs/`
    `tests/test_softmax_gs_metal_forward.py`

Contract already locked:
    The reference contribution tape returns final per-splat color weights. For
    any feature matrix:

    ```text
    rendered_color = weights @ features
    dL/dfeatures[k] = weights[k] * dL/drendered_color
    ```

    The bounded reference tape returns the top-K final contribution weights in
    ray order plus:

    ```text
    residual_weight = final_alpha - selected_weights.sum()
    ```

    For features in `[0, 1]`, dropping the residual contributors bounds every
    output-channel error by `residual_weight`.

Native tasks:

1. Cache or recompute per-contributor scalar tape rows:

   ```text
   input_T
   output_T
   input_absorbance
   effective_absorbance
   effective_past_absorbance
   prefix_weight_scale
   past_exponent
   output_past_exponent
   final_contribution_weight
   ```

2. Implement reverse propagation through the Softmax-GS scalar update into:

   ```text
   absorbance
   exponent
   depth
   beta/gamma if learned later
   ```

3. Reuse vanilla fast-mac local derivatives from exponent/absorbance into:

   ```text
   conic
   mean2d
   opacity
   color/features
   ```

4. Keep the overflow recompute route covered by tests until the native tape
   route covers the same support as the forward shader.

Tests:

```text
CPU tape reconstructs reference color
CPU tape gives exact color gradients
CPU bounded tape selects top-K weights in ray order
CPU bounded tape residual bounds unit-feature output error
MPS bounded tape matches reference for fast and forced-overflow tiles
MPS full-tape color backward matches reference for fast and forced-overflow tiles
MPS no-op parity with vanilla v5
MPS same-depth swapped-order behavior
MPS native fast and forced-overflow backward match Torch recompute
one-step enabled train smoke
```

Exit criterion:
    Initial gate met for fast and overflow tiles: enabled Softmax-GS backward no
    longer calls the Torch fallback on the standard tiny F3 smoke or
    forced-overflow smoke, and gradients match the reference within fast-mac
    tolerances. The bounded reference, fast/overflow Metal tape, tape-backed
    color-gradient contract, and full-tape selected scalar-gradient contract
    are met. Full exit still requires a K/residual policy that preserves train
    quality at useful scene density.

## Work Package E: Matched Dynamic-GS Smoke

Goal:
    Decide if Softmax-GS is worth a real renderer lane.

Baseline:
    The current fixed-512 dynamic-gsplat matched media comparator.

Compare:

- vanilla dynamic GS;
- Softmax-GS no-op parity row;
- Softmax-GS trainable/global row;
- optionally Softmax-GS lower primitive-count row.

Metrics:

```text
source/eval PSNR
source/eval SSIM
source/eval L1
alpha mean and alpha > 0.1 coverage
temporal flicker / frame-to-frame render delta
mean step time
backward time
raster time
media contact sheet
```

Promotion:
    Promote only if one of these holds:

```text
matched primitive count improves heldout/source quality without >1.25x step cost
matched quality holds with materially fewer primitives
temporal popping metric improves without quality loss
```

Do not promote from:

```text
source-view-only blur reduction
one pretty contact sheet
no heldout metric when heldout is available
quality gain with huge backward cost
```

## Work Package F: STAR CPU Diagnostic Only

Goal:
    Learn whether Softmax-GS affects the current STAR failure mode before
    porting it.

Files likely involved:
    `src/train/star_uvt_feature_tube_model.py`
    `research_experiments/star_uvt_feature_tubes/`

Tasks:

1. Add a slow dense CPU/Torch Softmax-GS option beside
   `dense_render_feature_tubes(...)`.
2. Run tiny same-depth two-tube order tests.
3. Run one sparse-1500 checkpoint diagnostic:

   ```text
   dense RGB PSNR
   forced-alpha PSNR
   alpha > 0.1 coverage
   target-background oracle
   feature/probe loss
   ```

Exit criterion:
    If alpha coverage and dense RGB remain in the known failed band, stop. Do
    not port STAR Softmax-GS to Metal.

## Work Package G: Notes And Baselines

Update only when measured:

- Append benchmark rows to `BASELINES.md` only for meaningful reruns.
- Add raw session notes to `agent_notes/loose_notes/`.
- Add a key learning only if the result changes the representation strategy.
- Update `EXPERIMENTS.md` if Softmax-GS becomes an active lane.

## Stop Conditions

Stop short-term Softmax-GS work if:

- dynamic-GS no-op parity is hard to establish;
- true-depth plumbing becomes a larger renderer refactor than expected;
- matched dynamic smoke improves source view but not heldout/view stability;
- STAR CPU diagnostic does not move coverage or dense RGB.

## Expected Outcome

The most likely useful output is not "Softmax-GS becomes the main renderer." It
is a clean answer to:

```text
Does overlap-aware splat compositing buy enough quality/stability to justify a
new fast-mac variant?
```

If yes, make it a renderer lane. If no, record it and keep attention on STAR
support and WorldFoam geometry.
