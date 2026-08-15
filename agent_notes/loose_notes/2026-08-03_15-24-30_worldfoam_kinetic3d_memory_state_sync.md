# WorldFoam kinetic-3D and memory-light state sync

## Context

This note reconciles the memory-light WorldFoam completion plan, the Lane 2
implementation README, and the dynamic-depth-order mathematician prompt after
the geometry and host-memory audits. It records current source truth; it does
not promote the native path or claim a publication run.

## Geometry truth

The fixed shared-SPD(4) world has a useful exact theorem, but it is not a
general-motion representation. In a fixed world-coordinate gauge, completing
the shared SPD(4) quadratic shows that every sliced 3D site shares one velocity.
After removing that common translation, sites are fixed, relative weights are
affine in time, and every candidate face normal is constant. This is exactly a
restricted translating-face kinetic family.

The rotating-face separation needs a gauge qualifier. A time-dependent global
spatial gauge can freeze one rotating face, so a two-site fixture is only a
strict separation in a fixed gauge (or modulo one declared common scene
gauge). One common gauge cannot generally freeze several independently
rotating faces. The intended formulation therefore keeps the gauged-camera
math: use one shared camera/scene gauge for bulk motion, then represent residual
cell motion with direct affine kinetic sites.

The direct CPU frontend uses

```text
p_i(t) = p_i0 + t v_i
w_i(t) = w_i0 + w_i1 t + w_i2 t^2.
```

For affine rays, its exact binary64-rational pair cut has coefficients
`A_ij(t), B_ij(t)` of degree at most two; adjacent-cut concurrence has degree at
most four. The source has exact fixed-time sparse lower-envelope words and
frame-independent site parameter bytes. A generic exact rational
square-free/Sturm isolation primitive through quartics and a guarded finite-cut
kinetic concurrence wrapper are implemented at CPU source scope. The complete
active-owner event set, half-open continuous chart compiler, native lowering,
kinetic geometry VJPs, and trainer integration are still open.

## Memory and work truth

The audited loss-only source ABI removes the discarded `B_p x K x 3`
prediction allocation. At `B_p=8192`, `K=8`, float32 RGB targets are 0.75 MiB;
an explicitly requested prediction is another optional 0.75 MiB. The selected
explicit-ray block is 1.5 MiB, and the audited `J=16` node-state plus cotangent
payload is about 4 MiB. These are logical tensor-payload formulas, not measured
native allocator peaks.

Material training no longer retains one compiled CPU atlas per spatial block
and performs no per-step CPU atlas compile. Its retained state is lightweight
compact topology, compact schedules, and owner bindings. The current session is
still a hand-built fixed-topology rectangular fixture, not the production
ragged paper sampler.

The verified sample-weight route has an `O(FJ)` common path with explicit
`O(F_fallback J^2)` row-local fallback. This is the desired structure: sample
the camera/time slice linearly because it is cheap, while keeping world-word
replay and the shared world reverse at compiler-node/event scale rather than
per-frame rasterization scale.

## Remaining completion boundary

1. Finish complete active-owner kinetic event enumeration and continuous chart
   emission using the guarded quartic primitive.
2. Lower kinetic topology and sparse geometry VJPs into the bounded native ABI.
3. Rebuild and run the native extension, then measure real allocator peaks and
   work scaling; source contracts alone do not prove runtime memory.
4. Replace the rectangular material fixture with the production ragged
   dataset/compiler/checkpoint/evaluator contract.
5. Expose `worldfoam_native4d` as a distinct unified paper-runner lane and run
   the fixed-duration frame-scaling ablation.

No MPS, CUDA, Metal build, publication-scale training, or broad benchmark was
run for this documentation sync.

## Documents synchronized

- `TODO/worldfoam_memory_light_native4d.md`
- `research_experiments/world_foam_lane2/README.md`
- `research_notes/worldfoam_paper/WORLD_FOAM_DYNAMIC_DEPTH_ORDER_MATHEMATICIAN_PROMPT.md`
