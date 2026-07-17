# Gauged UVT terminology correction: gauge domains, not weak charts

## Context

The user pushed on a real conceptual problem: if the theory is fiber bundles,
pullbacks, group/orbit gauges, and pushforwards, why are we saying "charts"?
Does that mean local fits are replacing richer math? Does throwing away charts
recover a better formulation?

## Current Model

The word "chart" is mathematically valid but semantically risky. Future work
should prefer:

```text
gauge domain with validity certificates
event-certified fiber cell
local trivialization
```

The invariant object remains:

```text
UVT trace = pi_* Gamma^* world_primitive
```

The local domain only says where a cheap expression and its certificates are
valid.

## Work Changed

Added:

```text
research_notes/gauged_uvt_trace_atlas/GAUGE_DOMAINS_NOT_CHARTS.md
```

Updated:

```text
research_notes/gauged_uvt_trace_atlas/README.md
research_notes/gauged_uvt_trace_atlas/GOAL_META_KEY_MATH.md
research_notes/gauged_uvt_trace_atlas/clean_thread_handoff/README.md
research_notes/gauged_uvt_trace_atlas/04_revolving_camera_atlas/README.md
```

The docs now state that "chart" means a local trivialization / gauge domain
certifying:

```text
projection regularity
trace error
support bounds
tile-time membership
depth/order behavior
interval gates
backward support
```

## Key Claim

Throwing away charts is only an improvement if the replacement still supplies
event cells for:

```text
denominator crossings
near/far crossings
image/tile boundary crossings
support birth/death
depth-order swaps
disocclusion boundaries
visibility ambiguity
```

Otherwise the renderer gives back the intended amortization of projection,
binning, support, ordering, and backward replay.

## Next Implication

For revolving cameras, prefer global group/orbit parameters and projective
rational gauges first. Split only at true events or certificate failures. The
implementation language can remain `ProjectiveTraceWindow` for now, but future
API names should trend toward `GaugeDomain`, `EventCell`, and `TraceCertificate`
when doing a cleanup pass.
