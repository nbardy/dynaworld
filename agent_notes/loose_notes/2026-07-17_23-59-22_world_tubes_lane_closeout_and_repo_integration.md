# World Tubes Lane Closeout And Repository Integration

Date: 2026-07-17 KST

## Context

This closeout was triggered by three questions:

1. Does the recent Gauged UVT/fiber-bundle work contain implementation worth
   preserving, or should the project use the other World Tubes/WorldFoam path?
2. Is the lane scientifically closed enough to stop expanding theory?
3. Why does the dirty repository appear to contain more than one million new
   lines, and what belongs in a clean `main`?

The live checkout was inspected rather than inferred from old notes. The
parent branch was three CamXTime commits ahead of local `main`; local `main`
was 114 fast-mac submodule-pointer commits ahead of `origin/main`. The parent
also had 125 tracked modifications and thousands of untracked files. The
fast-mac submodule had 18 tracked implementation modifications plus a large
untracked source-and-result surface.

## Observed Evidence

The following current-input checks passed before integration:

```text
190 passed, 1 skipped in 9.41s

projective_goal_final_completion_audit.py
    verified the saved final completion report against current inputs

paper_runner_table_report.py
    verified the saved nine-row paper-runner report

coffee_martini_matched_sweep_report.py
    verified the saved three-seed matched report
```

The focused pytest gate covered:

```text
World Tubes decisive demo
World Tubes visibility stress
World Tubes final completion audit
paper-runner table
Coffee Martini matched sweep
trainer registry and CLI
config/pipeline/sequence/temporal contracts
CamXTime multicamera loading
PowerFoam direct and dynamic Metal contracts
```

The matched Coffee Martini row is narrow but real:

```text
mean heldout PSNR
    World Tubes: 6.3863
    WorldFoam:    5.6311
    dynamic 3DGS: 4.9544

mean train-loop elapsed seconds
    World Tubes: 562.75
    WorldFoam:     22.46
    dynamic 3DGS:   2.43
```

This supports a one-split quality lead for World Tubes. It does not support a
general SOTA claim, and it exposes a large practical speed gap in the current
quality-oriented World Tubes policy.

## Current Model

### World Tubes

Status: promoted primary implementation and paper lane.

The useful durable object is:

```text
Trace_Gamma[rho] = pi_* Gamma^* rho
```

The implementation is not merely prose. It includes projective/rational trace
evaluation, continuous denominator certificates, support bounds, interval
atlas lowering, visibility strata/fallback accounting, exposure and rolling
shutter quadrature, Metal forward paths, and direct compiled-adjoint backward
paths.

A revolving camera is handled by the camera program and projective/orbit
gauge. Gauge choice can reduce trace complexity and chart count. It cannot
remove real geometric events such as denominator zeros, near-plane crossings,
support entry/exit, depth-order swaps, and disocclusion. Those event cells are
geometry, not evidence that the bundle math failed.

### WorldFoam

Status: parked retained-depth challenger.

WorldFoam is not redundant with World Tubes. The operator-order distinction is
real:

```text
World Tubes:
    early depth pushforward, then visibility certificates/repair

WorldFoam:
    retain the depth fiber, compose optical transfer, then push forward
```

Because visibility generally does not commute with depth marginalization,
WorldFoam remains scientifically useful. Current evidence does not justify
making it the default: its Metal bridge is scoped, its optical-transfer parity
is not fully native, and the one matched real-data split loses to World Tubes.

## Lane Decision

Close the open-ended Gauged UVT/fiber-bundle theory iteration.

Preserve and promote:

```text
camera-ray bundle invariant
projective/orbit gauge domains
continuous denominator certificates
compiled interval atlas
visibility strata and bounded fallback
finite exposure and rolling-shutter lowering
compiled direct adjoint
decisive-demo, stress, and paper-runner verifiers
```

Park, do not delete:

```text
WorldFoam retained-depth optical transfer
WorldFoam CPU monoid/VJP fixture
WorldFoam native shader variants and owner-run bridge
```

Only `world_foam_lane2_fused_slab_v0` remains a tracked native WorldFoam
implementation. The earlier base, fused-direct, and fused-CSR forks duplicated
large source surfaces and are ignored as local historical workspaces.

Stop by default:

```text
new umbrella bundle formalisms
more arbitrary chart terminology
single-split runner plumbing
local support/alpha sweeps without a new mechanism
bulk benchmark payloads in Git
```

## Reopen Conditions

Reopen World Tubes theory only if a replayable case falsifies one of:

```text
continuous denominator safety
support conservatism
visibility-stratum correctness
compiled-adjoint parity
orbit/exposure/rolling reuse
```

Reopen native World Tubes optimization only if phase timing shows bridge or
launch overhead is the limiting term after quality policy is fixed.

Reopen WorldFoam as an active implementation lane only if either:

```text
broader heldout camera/scene results beat World Tubes at matched budgets
native Metal optical transfer matches the retained-depth reference and shows a
measured end-to-end advantage
```

## Why The Repository Looked Like One Million Lines

The apparent size came from accumulated research state, not one million lines
of coherent application code.

Before cleanup, the parent had about 1.33 million untracked text lines:

```text
research_experiments   about 773k
standalone star_uvt    about 239k
research_notes         about 107k
agent_notes            about  91k
src                    about  53k
tests                  about  28k
```

The tracked parent tree was about 558k lines. The untracked total was inflated
by generated JSON sweeps, images/videos counted by line-oriented tools,
downloaded papers and extracted sources, duplicated experiment payloads, and a
standalone STAR results tree. Inside the fast-mac submodule, nearly all of the
multi-million-line apparent untracked count came from benchmark results,
including binary media interpreted by `wc`.

Repository policy after this closeout:

```text
track:
    source, shaders, tests, configs, curated notes, compact verifier fixtures

ignore locally:
    outputs, sweep result directories, media, downloaded paper corpora,
    browser-control state, superseded raw theory dumps
```

## Confidence

```text
High:
    bundle/gauge invariant, projective compiler correctness, current verifier
    contracts, distinction between World Tubes and WorldFoam

Medium:
    World Tubes is the best primary paper lane after broader evaluation

Low:
    current World Tubes quality policy is fast enough, one Coffee Martini split
    predicts broad scene performance, native WorldFoam will repay its complexity
```

## Decision Implication

The shortest next dependency chain is:

```text
clean and preserve source history
    -> broaden camera triplets and scenes
    -> compare matched quality/runtime/memory
    -> optimize only the measured bottleneck
```

No further theory folder is required to begin that work.
