# D-NeRF discontinuous-pose chart correction

The first D-NeRF adapter smoke established that the projective moving-camera
path executes, but a direct inspection of the official `bouncingballs` camera
poses invalidated the proposed single-chart publication policy.

The selected train and test sequences are matched in official time, but they
are not smooth camera trajectories. Consecutive pose changes approach `179°`;
the selected train poses accumulate roughly `2276°` of absolute rotation and
the test poses roughly `1771°`. Treating the ordered samples as one
first-order camera chart would therefore be mathematically unjustified even
though the trainer and renderer run.

The publication contract now declares this discontinuity in the manifest and
uses the conservative gauged fallback:

- `camera_sequence_mode=segmented`
- `segment_frames=1`
- one exact camera chart per posed sample
- explicit compiled-trace and chart-expansion diagnostics

The corrected two-step runtime artifact is:

`outputs/benchmarks/2026-07-22_dnerf_segmented_gauged_fallback_smoke_v3/`

It reports `64` active world tubes, `4` camera charts, and `256` compiled
traces for the four-frame smoke. This is intentionally not a sublinear camera
chart claim; it is the safe fallback for discontinuous official poses. The
bounded projective-chart contribution remains supported by the theorem and
same-representation scaling suites. The D-NeRF public row tests controlled
dynamic-scene breadth while exposing the fallback cost honestly.

The same pass also added explicit release of full rendered evaluation tensors
and the MPS cache before dynamic-3DGS evaluation. A 300-frame fixed-512
comparison process had exited after STAR media generation without producing a
report, consistent with retained cross-lane evaluation pressure. Both the
segmented D-NeRF smoke and the static Coffee two-step smoke pass with the
release boundary. The full fixed control still needs a clean rerun before the
memory diagnosis is considered closed.
