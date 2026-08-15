# WorldFoam sequential fixed-time full-geometry control

## Scope

Implemented the fair sequential same-representation control required by the
WorldFoam training-memory ablation. This is production code for an ablation
row, not a claim that WorldFoam has already fit within the target memory.

## Implementation

- Added `src/train/paper_kinetic_sequential_fixed_time_full_geometry_step.py`.
- Added a fixed-time physical-length geometry VJP to
  `research_experiments/world_foam_lane2/kinetic_stable_stratum_vjp.py`.
- Added an independent-reference CPU fake-native parity fixture in
  `research_experiments/world_foam_lane2/test_paper_kinetic_sequential_fixed_time_full_geometry_step.py`.

For each selected time, the control loads one selected-target frame, discovers
the exact fixed-time lower-envelope word for each ray, sends bounded track
blocks through the existing native P0 forward/full-VJP ABI, scatters material
bars, fences, copies only physical-length bars to the CPU, reverses them into
position/velocity/weight bars, and releases all frame-local state. It invokes
the supplied optimizer callback exactly once after the full gradient has been
assembled. The continuous kinetic compiler is never called by this control.

The receipt retains only an aggregate rolling digest and scalar counters. It
reports the cheap `F * 8` selected-time grid separately; expensive topology and
reverse scratch remain bounded by one frame/block. It also reports measured
gradient norms and exact lower-envelope/native/reverse interaction counts.

## Important evidence boundary

`native_callable_identity_digest` is intentionally only an in-process callable
identity. The fresh-process producer must separately record stable extension
binary/source hashes. Likewise, this control reports logical tensor-byte peaks;
the producer must record process RSS, MPS allocator/driver peaks, restart
parity, and target-loader provenance. Until the F=8/64/300 fresh-process rows
run, there is no empirical claim that WorldFoam fits the desired memory.

## Verification

No MPS workload, native build, or training run was launched.

```text
python compilation: passed
focused CPU gate: 20 passed
```

The CPU fixture uses multiple sites, rays, times, and owner runs; compares loss
and material/position/velocity/weight gradients against an independently
differentiated fixed-word physical-transfer reference; asserts every gradient
family is nonzero; and demonstrates one real manual SGD mutation callback.

## Remaining integration

The parallel ablation adapter owns fresh-process driver integration and must
bind this callable to the real selected-pixel target stream, native completion
fence, stable native/source identity, hardware/allocator receipts, and the
measured F=8/64/300 matrix. Verifier and acceptance JSON were deliberately not
edited here to avoid conflicting schema ownership.
