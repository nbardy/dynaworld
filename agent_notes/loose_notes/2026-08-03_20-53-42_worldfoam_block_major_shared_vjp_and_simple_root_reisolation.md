# WorldFoam block-major shared VJP and simple-root re-isolation

Date: 2026-08-03

## Why this pass mattered

The source tensors were frame-light, but the request-local lifecycle could
still rerun the expensive compiled node forward and ordered-word VJP once per
`K`-frame request. That hidden `ceil(F/K)` invocation factor would have broken
the central claim even though no persistent tensor carried an `F` axis.

The corrected lifecycle is spatial-bundle outer and temporal-chunk inner. Each
active native block is refreshed once, all requested observations stream into
its bounded node cotangents, its material-only ordered-word VJP runs once, its
compact bar scatters once into one union-local bar, and the bundle is then
released. The material path allocates no discarded `[J,W]` geometry-length
bar.

For heterogeneous blocks live together inside spatial bundle `q`, the honest
logical live-state estimate is

```text
M_live,q =
  16 S_union,q
  + sum_(b in q) (16 S_b + 32 R_b J_b)
  + max_(b in q) 16 S_b
  + O(1),

M_step_peak = max_q M_live,q.
```

The older `32 B_p J_max` node-state expression is valid only for a separately
proved sequential-single-native-block schedule. Persistent topology, binding,
and token bytes remain a sum over retained blocks and must be reported
separately from live scratch.

## CPU/fake-native evidence

`kinetic_ragged_paper_step_cpu_fake_native.py` and its behavioral tests prove:

- `K=1` and `K=4` produce the same loss and material bar as an independent
  direct-autograd oracle;
- increasing requested density from `F=5` to `F=41` changes only cheap sample
  work, not node-forward count, word-VJP count, word interactions, or retained
  runtime bytes;
- actual ranks `{3,4,5}` stay ragged, with no `J_max` padding or common time
  refinement;
- repeated site ids sum correctly into one caller-owned global `[S,4]` bar;
- incomplete coverage or stale provenance fails before optimizer authority;
  and
- sequential spatial bundles peak at the largest bundle, not their sum.

This is source/integration proof, not rebuilt Metal parity or allocator proof.

## Eventful geometry-update result

`kinetic_simple_root_reisolation.py` adds a restricted exact CPU certificate
for a directional geometry update across a multichart program. It reconstructs
the complete rooted and rootless predicate registry from every base owner word,
certifies disjoint singleton simple-root tubes, certifies every complement
root-free with exact tensor-Bernstein signs, rejects ray collapse and cut-
denominator collapse, re-isolates candidate endpoints, and exactly compares
left/right semantic owner words. A fixed-seed differential suite perturbs pair
and triple-event strata and requires every accepted candidate to match a fresh
exact compile.

Repeated roots, shared roots, persistent-zero predicates, endpoint events,
collapse, and ambiguous semantic changes fail closed. The reference is
conservative and not output-sensitive: registry construction is `O(U S R)`;
the tensor-Bernstein proof may take `O((K+M) 2^D)` leaves; endpoint isolation
uses `M` Sturm calls; and semantic validation uses `M+1` fixed-time hulls.
Warm affected-source repair and all derivatives through event times, chart
boundaries, rank, or compiler decisions remain open.

## Mathematical conclusion

No second Gaussian-style Schur complement is appropriate. Depth order is the
information WorldFoam exists to preserve. Its closure mechanism is instead:

```text
kinetic event/chart atlas
+ exact associative affine optical-transfer words
+ adaptive J-node temporal approximation
+ streamed residual-to-node reduction
+ one sparse node/world VJP per active compiled block and optimizer step.
```

The unavoidable output/camera/sample slice remains `Omega(PF)`. The defensible
claim is frame-density-independent expensive topology/word/world reverse work
at fixed physical interval, world, camera program, chart/rank tolerance, and
spatial coverage—not sublinear total image generation.

## Safety and literature boundary

No Metal, MPS, CUDA, extension build, or publication-scale training ran. The
literature search found static neural foams and kinetic-geometry precedents,
but no primary source that already combines dynamic kinetic cell/ray topology,
retained ordered optical transfer, a frame-density-independent temporal atlas,
and a reusable sparse cross-time adjoint. That is a search-based novelty
hypothesis, not proof of absence.
