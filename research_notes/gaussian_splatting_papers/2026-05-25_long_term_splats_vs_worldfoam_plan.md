# Long-Term Plan: Better Splats, STAR UVT, And WorldFoam

Date:
    2026-05-25
Updated:
    2026-05-28

Scope:
    Representation strategy after reading Softmax-GS and comparing it against
    the current STAR UVT / WorldFoam state.

## Current Strategic Plan: 2026-05-28

The plan is not "move to foam now" and not "port every Softmax-GS idea into
STAR." The plan is:

```text
short term: STAR visibility/support/composition gate
medium term: small representation tournament
long term: promote the family that wins heldout predictive quality per cost
```

Near-term mainline:
    STAR UVT stays the active dynamic world-token bridge because it already has
    the projective interval/trace-atlas math, first-class source-view trainer
    route, and practical direct-atomic feature path. Its blocker is now clear:
    support-changing rows can improve scalar/probe metrics, and the 2026-05-26
    binfix proves selected moving support was being incorrectly culled by the
    sparse binner. The open question has narrowed from "does support birth work
    at all?" to "does repaired local support become dense/world support, or do
    we still need visibility-prefix/compositing behavior?"

Softmax-GS role:
    Keep as a dynamic-GS renderer probe. It is valuable because it isolates
    overlap/order/compositing weakness in ordinary splats. It should not be
    ported into STAR until STAR has enough target support that overlap/order is
    visibly the remaining failure, or until dynamic GS gets a repeated heldout
    win from Softmax-GS at matched budget.

WorldFoam role:
    Keep as the serious challenger and possible future primitive family.
    WorldFoam may be the cleaner answer if the core failure is primitive
    ownership/topology rather than renderer-local composition. But it becomes
    mainline only after a matched row beats STAR/dynamic splats on the same
    clip, split, resolution, frame count, wall-clock budget, and heldout
    metrics.

The next decisive long-term artifact should be a tournament table, not another
isolated speed gate:

```text
dynamic GS baseline
dynamic GS + Softmax-GS only if repeat evidence improves
STAR UVT selected support/composition route
WorldFoam selected Metal route
```

Promotion rule:
    A representation wins by heldout predictive behavior, rate/quality, stable
    trainability, and clean export. Source-view media, shader speed, or elegant
    geometry are supporting evidence only.

STAR update after the binfix train:
    The immediate STAR mainline was strengthened, not closed. The repaired
    targetarea2 binfix row passes fixed-bin with zero overflow, max tile count
    `110/128`, total loss `0.889263 -> 0.863064`, support-target-area loss
    `0.253626 -> 0.217254`, and selected-patch normal/forced/oracle PSNR
    `6.644/19.452/26.994`. This is real local support evidence. It is not yet
    a representation promotion because dense transfer is only partial:
    whole-frame normal/forced/oracle PSNR is `7.269/14.736/21.439`, alpha
    `>0.1` is `75.4%`, and raw opacity bias only reaches `8.039` PSNR. This
    strengthens STAR enough to keep it as the short-term mainline, but it also
    said the next work should test prefix-weight/final-alpha composition, not
    another local support-target loss. That test is now measured: the 50-step
    prefix-alpha row learns local contribution (`0.4114 -> 0.4419` selected
    weight, `0.4456 -> 0.4751` final alpha) and passes fixed-bin, but dense
    transfer is essentially flat (`7.262/14.732/21.438`, alpha `>0.1` `75.4%`).
    The prefix tape still confirms selected born tubes are not meaningfully
    hidden on selected target rays (`93.8%` selected weight share, `96.9%` top
    selected, `0.8%` prefix-hidden). STAR therefore needs broader ownership/
    coverage or a different support distribution before this becomes a
    representation-promotion row.

## Immediate Strategic Answer

Do better splats now, keep STAR UVT as the main dynamic world-token bridge,
and keep WorldFoam as the serious challenger. "Better splats" now means
support-changing STAR work plus disciplined dynamic-GS probes, not a broad
Softmax-GS port. Do not switch the project to foam because Softmax-GS exposed a
splat weakness; make foam win a matched representation tournament first.

Concrete next move:

```text
record STAR 1300->1500 checkpoint ladder as regenerated
record K=8/r64/o0.4 50-step birth/split as a useful failed gate
record the cap-pressure reduction: n16 nearly clears but still overflows,
n8/r40/o0.4 is the current cap-128-safe support seed
record the longer safe-row split: 90-step checkpointselect passes, 100-step
overrun fails by a late objective jump
record the checkpoint-aware tail schedule: it makes the 100-step row pass but
does not beat the 90-step checkpoint or move dense support
record the allocation follow-up: uniform n16/K16 and r32 still hit the same
two-tile cap wall; K=12/n12 is cap-safe but not a selected-checkpoint win
record the cap-aware support bridge: tile slack alone fails, guarded repair
passes K16/n16 fixed-bin but only nudges dense support
record the residual-cap-slack scorer: scalar objective improves slightly, dense
support stays flat
record the footprint-residual scorer: scalar endpoint improves again, dense
support still stays flat
record target-grid feature init: content/oracle improves slightly, alpha
coverage stays flat
record support-target alpha: pointwise target alpha learns but only nudges dense
support
record support-target-area patches: local patch composition learns but lands on
the same support plateau
record the 2026-05-26 STAR UVT binfix: selected moving tubes were being culled
by conservative-bounds bookkeeping; repaired bounds make selected-only alpha
nonzero and the first targetarea2 binfix rerun passes fixed-bin
record the dense binfix diagnostic: dense normal PSNR rises to 7.269 and
alpha>0.1 to 75.4%, but forced-alpha/oracle remain 14.736/21.439 and raw
opacity bias reaches only 8.039 PSNR
record the prefix tape diagnostic: selected support owns sampled target rays
but final alpha is still weak, so prefix weight/final alpha was the right next
thing to test rather than hidden-support ordering
record the prefix-alpha train: local selected contribution rises, but dense
support is flat against binfix, so do not repeat alpha pressure without changing
ownership/sampling/coverage
then decide between a stronger visibility support bridge, learned dynamic-GS
Softmax-GS, or the first small representation tournament
```

Current stop rule:
    No Softmax-GS STAR/WorldFoam engineering until a repeatable dynamic-GS
    heldout/rate win exists, or until STAR support has enough coverage that
    overlap/order is clearly the remaining bottleneck.

## North-Star Question

Which representation should carry DynaWorld's world asset over the next phase?

Candidate families:

1. Better splats:
   dynamic GS, Softmax-GS-style compositing, improved fast-mac renderers.
2. STAR UVT:
   time-tubed splats, feature tubes, projective interval trace atlas, support
   birth/split and visibility-aware objectives.
3. WorldFoam:
   bounded cells / ray-cell intervals / surface-detail sites as a deeper
   geometry primitive.

The answer should not be aesthetic. It must be decided by heldout predictive
behavior, rate/parameter efficiency, export purity, and trainability.

## Strategic Read

Softmax-GS patches a real splat weakness: fuzzy overlapping Gaussians rely on
order-sensitive alpha-over compositing. WorldFoam attacks a related issue more
structurally by replacing overlapping fuzzy blobs with bounded spatial cells.

That does not mean WorldFoam is automatically the main path. A deeper primitive
can be more correct and still lose if it is too hard to train, too brittle to
initialize, too slow to backprop, or not yet wired to the DynaWorld data
contract.

Current stance:

```text
short-term mainline: better splats / STAR UVT
medium-term challenger: WorldFoam
long-term likely winner: whichever proves heldout world behavior at scale
```

Status update:
    The first Softmax-GS dynamic-splat implementation evidence is positive but
    narrow. The forward shader now fixes the exact same-depth order artifact on
    MPS, the no-op train route still runs, and enabled tiny train smokes now
    pass through native fast-tile and overflow recompute backward. The
    reference has an executable contribution-tape/color-gradient contract, and
    the native backward matches the Torch recompute reference on tiny MPS
    projected cases including depth gradients. Matched 64px/4f/128-splat
    10-step offline W&B diagnostics with media now exist. The fresh
    post-overflow 10-step run ended enabled `0.4198` vs no-op `0.4456`, but a
    cleaner seeded 50-step source-view row ended no-op `0.1467` vs enabled
    `0.1512`. This is not representation evidence and does not promote
    Softmax-GS. The reference now has a bounded top-K tape contract with a
    residual error bound, and `v5_softmax_gs` has fast/overflow Metal tape ABI
    coverage. Backward consumes the tape for color plus selected
    geometry/opacity/depth gradients when `softmax_gs_tape_k > 0`. Full-tape
    scalar tests are exact in fast and forced-overflow modes; bounded K is
    approximate. The K=8 50-step row is too lossy (`0.2026` final), while K=16
    recovers the seeded no-op/recompute bracket (`0.1472` final, offline run
    `wandb/offline-run-20260525_204816-oip27eka`). K=32 does not improve the
    tiny endpoint (`0.1588` final, offline run
    `wandb/offline-run-20260525_205435-wy8r4v9l`). The first tiny
    64px/4f/128-splat RGB-pyramid multicam row is now positive on heldout:
    no-op reaches heldout PSNR/SSIM `4.7369/0.0503`, enabled K=16 reaches
    `11.7255/0.0794`, while final train loss is tied (`0.2261` vs `0.2262`).
    The first primitive-count repeat does not preserve the heldout-PSNR lift:
    at 64px/4f/512 splats, no-op reaches heldout `12.5002/0.0817` while enabled
    K=16 reaches `11.8847/0.0950` despite a better final train loss
    (`0.2378` vs `0.2511`). The old 128px/8f MPS model-forward crash is now
    localized to large-memory `nn.MultiheadAttention` and fixed with a manual
    MPS cross-attention fallback; unstrided 128px/16f forward works, but full
    training was interrupted after 3/20 steps at 9:47 as locally impractical.
    The practical 128px/16f/512 stride16 repeat is mixed: no-op reaches heldout
    `12.1234/0.1244`, enabled K=16 reaches `12.2092/0.1088`, and enabled also
    loses train-view metrics. K=16 bounded-tape residual is small on the trained
    512-splat row (heldout residual/alpha mean/p99 `0.001930/0.012332`), so the
    negative/mixed repeats are not just tape-truncation artifacts. This keeps
    Softmax-GS as a renderer probe, not a representation-direction promotion.
    STAR support is now the active near-term path, and the first long support
    row has resolved into evidence rather than speculation. The selected
    `K=8/r64/o0.4` 50-step birth/split continuation now runs from the
    regenerated sparse-forward 1500 checkpoint and improves total/feature/probe
    losses, but fails the cap-128 fixed-bin contract with `277` overflowed
    tiles and max tile count `146/128`. This keeps the representation decision
    on STAR support. The cap-pressure reduction follow-up says the budget
    boundary is tight: `n16/r48` and `n16/r40` improve losses but still overflow
    two tiles (`131/128`), while `n8/r40/o0.4` passes fixed-bin (`0` overflow,
    max tile `123/128`) and still improves loss/probe plus dense support over
    the regenerated 1500 checkpoint. The longer safe-row test selects the
    90-step checkpoint (`0.754568 -> 0.747006` loss, feature
    `0.608402 -> 0.606764`, RGB-probe PSNR `24.372 -> 24.552`, zero overflow,
    max tile `122/128`) and rejects the 100-step endpoint, which remains
    fixed-bin but regresses after late jumps at global steps `1590` and `1594`.
    The checkpoint-aware 100-step schedule (`lr=0.001` until global `1588`,
    then `0.00025`) passes with zero overflow and final loss `0.749454`, but
    remains worse than the selected 90-step checkpoint and has the same dense
    support profile (`6.462/14.012/21.578` normal/forced/oracle PSNR). This is
    a valid support seed, not a visual-quality closeout. The allocation
    follow-up sharpens that read: uniform `n16`, one-tube-per-center
    `K=16/n16`, and `K=16/n16/r32` all still fail cap-128 by two tiles, while
    `K=12/n12/r40/o0.4` is cap-safe but does not beat the selected `K=8/n8`
    90-step checkpoint on objective/feature loss and does not improve dense
    forced-alpha/oracle support. The next concrete work is a smarter
    visibility/support bridge before changing the math or moving to
    Softmax-GS/WorldFoam. The first cap-aware support bridge is now measured:
    cap-slack target scoring alone still hits the same two-tile overflow,
    exact-fit repair drifts to one final overflow tile, and guarded repair
    (`K=16/n16/r40/o0.4`, guard `2`) passes fixed-bin with max `127/128`.
    Its dense support read is only `6.486/14.021/21.571`
    normal/forced/oracle PSNR, a tiny nudge over `K=12/n12`, so it promotes
    guarded repair as a cap-safety primitive rather than a representation
    decision. The first residual-cap-slack scorer is also measured: it selects
    high-residual/low-alpha points and improves scalar objective/probe slightly,
    but dense support remains flat (`6.486/14.019/21.579`). This keeps the next
    concrete work on a stronger model handoff, not a representation switch. The
    footprint-aware residual scorer has now been measured too: it gives the best
    K16 scalar row (`0.752912 -> 0.748672`) but dense support remains flat
    (`6.481/14.021/21.576`). The first target-grid feature-init handoff is a
    small positive (`0.752454 -> 0.748504`, dense `6.488/14.054/21.629`), but
    alpha coverage stays flat, so this narrows the next work to alpha/
    composition or visibility-prefix behavior. The first direct support-target
    alpha bridge confirms the same read: the sparse pointwise alpha objective
    learns (`0.492962 -> 0.478448`) and nudges dense support to
    `6.508/14.084/21.626`, alpha `>0.1` `0.657`, but does not materially close
    the forced-alpha/oracle gap. The support-target-area 2x2 patch bridge is
    cheaper and learns locally (`0.597970 -> 0.581641`) but lands on the same
    plateau (`6.507/14.085/21.627`, alpha `>0.1` `0.657`) while weakening
    feature loss. The 2026-05-26 binner repair changes this from a pure plateau
    to a partial dense support gain: selected-only alpha becomes nonzero, the
    first targetarea2 binfix row passes fixed-bin with max tile `110/128`, and
    dense support improves to `7.269/14.736/21.439` normal/forced/oracle PSNR.
    But the remaining gap is still too large, and raw opacity bias only reaches
    `8.039` PSNR. Prefix tape shows selected support is present and dominant
    locally (`93.1%` weight share, `95.7%` top selected). The prefix-alpha
    follow-up learns local contribution but leaves dense support flat against
    binfix, so the next STAR move needs broader ownership/coverage or a
    different support distribution, not another pointwise/small-patch/prefix
    pressure variant.

## Work To Do Now

This is the practical ordering after the Softmax-GS read and the current STAR /
WorldFoam state:

1. Treat the current Softmax-GS renderer probe as characterized enough for
   strategy. K=16 is viable, K=8 is too lossy, K=32 does not improve the tiny
   endpoint, and repeat/scale quality evidence is mixed.
2. Treat the selected STAR checkpoint ladder as rebuilt and the 50-step
   `K=8/r64/o0.4` support continuation as a measured failed gate. This is
   still the immediate evidence because STAR's known failure is
   support/coverage/composition, not merely alpha-over ordering.
3. Continue STAR UVT support/projective-interval work as the main splat-time
   representation lane, because its current blocker is support/coverage and
   heldout world behavior, not only final compositing. The immediate concrete
   STAR action is now a smarter support-selection bridge around the cap-safe
   `K=8/n8/r40/o0.4` row. The checkpoint-aware schedule has now been tried as a
   tail stabilizer and is only a stability aid. `n16` kept the benefit but
   missed fixed-bin by two tiles, so radius shrinkage alone is not the main
   lever; 90 steps improves the objective, constant-LR 100 steps over-runs into
   a late jump, and scheduled 100 steps passes without moving dense support.
   Uniform allocation and `K=16/n16` show the two-tile cap wall is not just
   proportional center packing; `K=12/n12` shows a slightly larger cap-safe
   support set can fit, but it does not change the selected checkpoint. The
   first tile-cap-slack bridge says slack-aware target placement still needs a
   guarded repair pass to stay fixed-bin, and even then it only nudges dense
   support. Pointwise residual-cap-slack scoring improves the scalar endpoint but
   not dense support; footprint-aware residual scoring improves the scalar
   endpoint again but still does not move dense support. Target-grid feature
   init improves the born support's content/oracle read slightly, but does not
   move alpha coverage. The first support-target alpha objective learns locally
   but only nudges dense support; the first support-target-area patch bridge is
   cheaper but lands on the same plateau. The binner repair gives partial dense
   transfer (`7.269/14.736/21.439`), and prefix tape shows selected support
   owns sampled target rays. Prefix-alpha training then raises local selected
   contribution but not dense support, so the next STAR gate should broaden
   support ownership/coverage or change sampling rather than repeat local
   target/prefix pressure.
   Cap-256 remains only a
   diagnostic of budget pressure, not a mainline promotion.
4. Keep WorldFoam as the serious challenger, but judge it by heldout quality
   and trainability rows, not by isolated shader speed wins.
5. Build a small representation tournament before declaring any future
   mainline switch.

Near-term answer:
    Do better splats now. Do not move wholesale to WorldFoam yet. Do not port
    Softmax-GS into STAR until the dynamic-GS evidence becomes repeatable or
    STAR support coverage is no longer the primary failure.

Medium-term answer:
    STAR UVT remains the main dynamic world-token bridge if support-changing
    work starts moving dense RGB and heldout metrics. WorldFoam becomes the
    mainline only if it wins the same split/resolution/budget comparison.

Long-term answer:
    Foam may be the better primitive family, but only if it trains into a
    compact predictive asset under the DynaWorld contract. Elegance is not
    enough.

## Representation Bets

### Bet A: Better Dynamic Splats

Claim:
    Keep classic/dynamic Gaussian splats, but improve compositing, depth,
    support, and renderer speed.

Why it could win:

- fastest iteration loop;
- easiest baseline compatibility;
- easiest to plug into existing TokenGS/dynamic-GS trainers;
- Softmax-GS may improve sparse boundary quality without changing the model;
- good for quick heldout-camera probes.

Why it could lose:

- fuzzy tails and order artifacts require endless renderer patches;
- high-quality dynamic scenes may need too many primitives;
- source-view overfit can look better than heldout behavior;
- splat support remains a learned accident rather than a structural surface.

Promotion evidence:

```text
heldout camera quality beats current dynamic-gsplat rows
parameter count drops at matched quality
temporal popping metrics improve
runtime remains close to current fast-mac envelope
```

Kill / deprioritize evidence:

```text
Softmax-GS and related compositing patches improve only source-view media
quality gains require big backward/tape cost
heldout remains below dynamic/foam alternatives
```

### Bet B: STAR UVT As The Splat-Time Mainline

Claim:
    STAR UVT remains the best bridge between splats and dynamic world tokens
    because it represents time as continuous screen-time traces and can share
    work through projective interval atlases.

Why it could win:

- source-view RGB STAR has strong overfit quality;
- feature-tube route has fast practical sparse-forward/batched VJP surfaces;
- projective interval trace atlas aligns with camera-ray bundle math;
- support birth/split is a real primitive for changing coverage;
- it keeps the fixed-rasterizer/export contract alive.

Why it could lose:

- current feature route is blocked by coverage/visibility/composition;
- target-grid/probe success does not automatically become dense RGB quality;
- support fixes can trade coverage against oracle quality;
- native visual VJP work can become shader-churn without visual gain.

Near-term decisive gates:

```text
support-changing STAR rows must lift dense RGB, not only alpha coverage
projective interval route must scale beyond tiny F3/source-view smokes
mixed same-view + heldout training must prove STAR world-token relevance
```

Role of Softmax-GS:
    Later quality/stability lever, not first support bridge. Apply only after
    STAR has enough target support that overlap/order is visibly the bottleneck.

### Bet C: WorldFoam / Brilliant Foam

Claim:
    A bounded-cell/ray-cell representation is a better long-term world primitive
    than fuzzy splats.

Why it could win:

- explicit spatial cells reduce arbitrary overlapping-splat ownership;
- ray-cell intervals match real visibility and surfaces more naturally;
- surface/detail sites can express materials at cell boundaries;
- topology/adjacency gives a stronger geometry object than a bag of splats.

Why it could lose:

- topology, adjacency, and grow/prune are hard to keep stable;
- local Metal is not full official PowerFoam parity;
- quality can be blocked by initialization and heldout ray support;
- dynamic video adaptation is a separate problem from static posed-camera foam;
- WorldFoam can be mathematically elegant but still too expensive to train.

Promotion evidence:

```text
heldout-camera quality beats dynamic splats and STAR on same split
optimizer steps are stable from clean/reasonable init
raytrace/Metal gates pass at target resolution
export asset is compact and pure
quality scales with more data instead of only hand-picked scenes
```

Kill / park evidence:

```text
quality remains below dynamic splats at matched budget
heldout improvements depend on camera/ray convention quirks
topology refresh dominates training
official parity remains unavailable for claims that need it
```

## Key Decision: Do We Move To Foam?

Not yet as the mainline.

Move WorldFoam to mainline only after a direct head-to-head row clears:

```text
same dataset split
same render resolution / frame count
same train-camera and heldout-camera protocol
comparable wall-clock budget
same metric table in BASELINES.md
```

Until that row exists:

- better splats/STAR stay the fast iteration lane;
- WorldFoam stays the future-primitive challenger;
- Softmax-GS stays a low-cost splat renderer probe.

## Long-Term Milestones

### Milestone 1: Clean Representation Tournament

Build one small but honest tournament:

```text
dynamic GS baseline
dynamic GS + Softmax-GS if short-term probe is positive
STAR UVT selected support route
WorldFoam selected local Metal route
```

All rows must use:

```text
same clip/split
same train/heldout camera semantics
same frame count
same resolution
separate source-view and heldout-view metrics
wall-clock and step timing
media artifacts
```

Outcome:
    Decide which representation gets the next serious engineering block.

### Milestone 2: Rate/Quality Frontier

Do not compare only PSNR. Compare:

```text
export size
primitive/cell count
train step time
render time
heldout PSNR/SSIM/L1
temporal stability
coverage/support diagnostics
```

The desired representation is not the prettiest one at unlimited capacity. It
is the best minimal predictive asset under the DynaWorld contract.

### Milestone 3: Mixed Same-View + Heldout Training

Representation choice is incomplete until it runs through the mixed data
contract:

```text
same-view scale pretraining
multicam heldout loss
separate logs
same exported asset
no query-camera learned branch
```

This is where many attractive renderer ideas will fail if they only improve
source-view reconstruction.

### Milestone 4: Export And Browser Contract

A future primitive must export cleanly:

```text
asset = world representation
render(asset, camera, time) -> image
no source frames
no teacher tensors
no hidden per-frame cache
known renderer version
```

Splats are already easy here. STAR UVT is plausible. WorldFoam needs its
topology/adjacency/feature state exported cleanly.

## What To Avoid

Do not:

- jump to WorldFoam because it feels more physically elegant;
- keep patching splats forever if heldout quality stays flat;
- promote Softmax-GS from source-view media alone;
- compare WorldFoam speed gates against STAR quality rows;
- call projective interval cache wins a visual-quality solution;
- let shader speed work substitute for representation selection.

## Recommended Allocation

For the next phase:

```text
50% STAR support / mixed-training / projective interval quality gates
25% dynamic-GS + Softmax-GS cheap renderer probe
25% WorldFoam heldout-quality challenger gate
```

Adjust only after measured evidence:

- If Softmax-GS improves dynamic heldout quality cheaply, spend more on better
  splats.
- If WorldFoam beats heldout at matched budget, move it from challenger to
  mainline.
- If STAR support finally moves dense RGB quality, keep STAR as the dynamic
  world-token bridge and use Softmax-GS only as a later compositing option.

## Final Position

Softmax-GS suggests that some splat weaknesses can be patched. WorldFoam
suggests that some splat weaknesses are symptoms of the wrong primitive.

The correct DynaWorld strategy is not to choose by philosophy today. It is to
make the tournament small, honest, and brutal:

```text
heldout behavior first
rate/quality second
speed third
source-view prettiness last
```

Until WorldFoam wins that tournament, the project should continue improving
splats/STAR while keeping foam alive as the serious long-term challenger.
