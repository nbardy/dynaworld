# World Foam Auto Selector Reflection

We stopped this chunk after adding a Python-side auto-selector for the two
surviving endpoint-record-delta-replace-coeff16 framegroup16 layouts. This was
not a new Metal ABI or new fused shader. The selector chooses the packed
framegroup16 layout for `frame_count <= 64` and the i16x3 framegroup16 layout
above that.

Changed files:

- `research_experiments/world_foam_lane2/train_eval_owner_run_tape.py`
- `research_experiments/world_foam_lane2/compare_delta_framegroup_i16x3_packed_train_eval.py`
- `research_experiments/world_foam_lane2/test_compare_delta_framegroup_i16x3_packed_train_eval.py`

Verification that completed:

- `py_compile` passed for the three touched Python files.
- The focused unittest passed: `Ran 5 tests ... OK`.
- Runtime smoke passed for auto mode at 16f and 128f:
  - 16f resolved to `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`.
  - 128f resolved to `endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse`.
  - 8x frame-count increase gave total-step scale `0.980x`, backward scale `1.042x`, storage scale `1.066x`.
  - Heldout PSNR stayed in the same smoke range: `15.463` at 16f, `15.486` at 128f.

Important caveat: the absolute smoke timings were cold and not competitive with
the cleaner promoted low-ms fused-lossreduce rows. The saved smoke reported
`356.014 ms` total / `310.989 ms` backward at 16f and `348.917 ms` total /
`323.937 ms` backward at 128f. Treat this as selector wiring proof and
sublinear-shape evidence, not a speed promotion.

The paired i16x3-vs-packed-vs-auto compare was interrupted before completion.
The final JSON was not written. The partial JSON only preserved mode topology
and selector resolution state after failure; it nulled the timing rows, so it
is not a promotable oracle artifact.

Reflection:

- The auto-selector idea is plausible as an audit harness: packed wins the
  small-frame side in prior partial observations, while i16x3 was the more
  stable high-frame fallback.
- It does not by itself close the STAR-UVT gap. It chooses between two World
  Foam tape layouts; it does not remove the remaining replay/dispatch overhead.
- The right next measurement, if this lane continues, is a fully completed
  paired compare with enough warmup and interleaved mode order, followed by a
  robust timing verifier. Without that, the selector should remain experimental.
