# fast-mac v5 training parity fix

## Context

The trigger was `third_party/fast-mac-gsplat/docs/v5_dynaworld_train_parity_report.md`.
That report showed the native Torch+Metal v5 renderer was much faster than dense
Torch on the tiny Dynaworld baseline, but trained badly:

```text
dense    eval_loss ~= 0.065807
taichi   eval_loss ~= 0.066723
fast_mac eval_loss ~= 0.176579
```

This mattered because first-step Taichi and fast-mac losses matched, so the
failure looked like a train-time/backward issue rather than a basic projection
or API-shape issue.

## Current Model

The v5 Dynaworld adapter was basically calling the right API. The main break was
inside the v5 Metal backward implementation for saturated/crowded tiles.

Confidence: high for the identified kernel bug; medium for "all training parity
issues are solved", because longer real-trace training still needs coverage.

## Evidence

The v5 bundled tiny reference check originally passed for `B=1/B=2, G=4`.
That was not enough coverage. Larger projected tests showed:

- spread / low-overlap cases matched
- crowded or opaque cases matched forward but had wrong gradients
- finite differences disagreed with v5 backward under saturation
- Taichi finite differences matched its own backward in the same setup

The important post-fix validation:

```text
saturated image max error: 2.086162567138672e-07
saturated means grad max error: 2.3283064365386963e-10
saturated conics grad max error: 1.1920928955078125e-07
saturated colors grad max error: 3.725290298461914e-09
saturated opacities grad max error: 9.313225746154785e-10
```

Adapter-level v5 vs Taichi after the fix, crowded case:

```text
CASE crowded B 2 G 64
loss taichi/fast 0.7882294058799744 0.7882294058799744
image max=2.98023e-07
xyz max=2.23517e-08
scales max=5.32717e-07
quats max=7.12462e-08
opacity max=2.79397e-09
rgb max=1.86265e-09
```

Tiny training control after the fix:

```text
FINAL_METRICS Loss=0.069294 Eval/Loss=0.070275 Eval/L1=0.068052 Eval/MSE=0.011115 Eval/SSIM=0.478317 Eval/PSNR=19.541011
```

That does not exactly reproduce the old dense/Taichi run, but it moves v5 from
catastrophic divergence into the same small-baseline band.

## Root Cause

File:

```text
third_party/fast-mac-gsplat/variants/v5/csrc/metal/gsplat_v5_kernels.metal
```

Kernels:

```text
tile_fast_backward_saved(...)
tile_overflow_backward(...)
```

The reverse backward loop used each pixel's `end_i` as the loop bound while the
loop body contained `threadgroup_barrier()`. `end_i` is per pixel because forward
compositing can stop when transmittance falls below the threshold. In crowded or
opaque tiles, different pixels stop at different Gaussian indices.

That means different lanes in the same tile can execute different numbers of
barriers. This is invalid cooperative threadgroup control. The symptom was not a
clean crash; it was corrupted gradients in exactly the saturated scenes that
training can hit.

The fix:

```text
fast path:     for chunk_end in uniform tile stop_count
overflow path: for chunk_end in uniform tile count
per-pixel math: keep `global_i < end_i` as a contribution mask
```

This preserves the early-stop math while making barrier control uniform.

## Branches Considered

Hypothesis: Dynaworld passed tensors with the wrong shape/layout.

Status: weakened. The adapter calls v5 with `[B,G,2/3]` projected tensors and
rank depths as expected. Forward parity and first-step loss parity made a gross
API mismatch unlikely.

Hypothesis: v5 forward math differed from Taichi/dense.

Status: weakened. Forward matched in spread and crowded tests. The training
failure appeared after optimization.

Hypothesis: v5 backward mishandled sort/unsort or saved IDs.

Status: partially checked. Saved ID behavior was not the root issue for the
failing tests. The wrong gradients were reproduced in saturated projected cases
and fixed by the barrier-uniform loop.

Hypothesis: overflow path caused mixed fast/slow gradient mismatch.

Status: not the primary bug. Both fast and overflow backward had the same
barrier-control pattern and were patched the same way.

## Falsification Tests To Keep

1. Run the v5 reference check after every Metal backward change:

```bash
cd third_party/fast-mac-gsplat/variants/v5
python tests/reference_check.py
```

2. Keep a saturated many-splat projected reference in that test. The original
`G=4` tiny case missed the bug.

3. Keep an adapter-level v5-vs-Taichi gradient check for crowded projected
inputs. This catches bridge/projection mistakes that a pure projected renderer
test cannot see.

4. Periodically rerun the 100-step tiny Dynaworld control with dense, Taichi,
and fast-mac under a pinned seed in one clean machine state.

## Decision Implications

v5 should no longer be rejected as "fast but not trainable" based on the old
report. It is now a viable native-batch Metal renderer candidate.

Do not generalize too far from the tiny fix:

- dense remains the correctness baseline
- Taichi remains the compatibility baseline
- v5 needs real-trace and longer-run coverage before becoming the universal
  default
- v3 may still be preferable for some B=1 large-scene cases

The durable engineering lesson is that renderer correctness tests need
saturated, crowded, early-stop-heavy scenes. Low-overlap tiny cases can pass
while the real training regime is broken.
