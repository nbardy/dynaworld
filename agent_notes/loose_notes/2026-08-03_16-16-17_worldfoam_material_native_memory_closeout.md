# WorldFoam material-native memory closeout

## Scope

This source/CPU pass tightened the fixed-topology material-training lifecycle.
It did not build Metal, run MPS/CUDA, launch training, or establish allocator
peak measurements.

## Changes

- Added a material-only native reverse ABI. The owner-material binding keeps
  loss, node cotangents, and compact RGBA bars, but skips Mobius bars
  `[I_b,4]`, boundary bars `[B_b,5]`, and final geometry gradients `[S_b,5]`.
  Strict/evaluation bindings retain the full geometry path.
- Added a session-owned, fail-closed topology-token cache keyed by immutable
  program, binding, schedule, device, and native-implementation identity. A
  cold two-block step prepares two tokens; a matching later step prepares none.
  Live material/world tokens still refresh every block and step, and cache
  residency is bounded by spatial block count.
- Added a separate target-only staging API and routed only owner-material
  training through it. One bounded `[B_p,1,6]` reference row validates fixed
  affine camera coefficients exactly, then hot blocks carry only
  `[B_p,K,3]` targets. Moving cameras, camera gradients, nonzero slopes, and
  coefficient mismatches fail closed. Strict/evaluation and piecewise paths
  keep their explicit-ray route.
- Repaired the piecewise adapter's stale import of the renamed public native-op
  resolver, found by the combined integration gate.
- Updated host accounting to separate strict/evaluation and material payloads,
  include the bounded topology cache, and retain the distinction between
  source tensor lower bounds and measured allocator peaks.

## Exact source-level savings

Material-only reverse omits `16 I_b + 20 B_b + 20 S_b` bytes per active block.
Target-only staging changes the hot sample payload from `36 B_p K` to
`12 B_p K` bytes, saving `24 B_p K` bytes. At `B_p=8192,K=8`, this is
`2.25 MiB -> 0.75 MiB`, a `1.5 MiB` reduction. The audited material sample
phase is now

```text
76 S_b + 112 B_p + 40 B_b + 20 I_b + 12 W_b
+ 32 B_p J + 12 B_p K + 4 K J bytes,
```

excluding allocator reservations, Python objects, command buffers, optimizer
state, and unexposed native temporaries.

## Verification

- Combined target-staging, native adapter, material trainer, piecewise adapter,
  native source verifier, and host-memory gate: `72 passed, 11 subtests passed`.
- Kinetic frontend, exact polynomial roots, sparse owner compiler, route-cost,
  and frame-density gates: `42 passed, 16 subtests passed`.

The native extension remains unbuilt and unmeasured. The general dynamic route
still needs complete active-owner kinetic event enumeration, continuous chart
emission, native kinetic lowering/VJPs, and production runner integration.
