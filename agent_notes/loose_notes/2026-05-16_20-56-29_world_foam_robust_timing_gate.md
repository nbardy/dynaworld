# WorldFoam Robust Timing Gate

## Context

The framegroup16 fused-MSE shader lane had two conflicting reads:

- the accepted loss-reduced render32/site12 artifact was clean and sublinear;
- the later post-owner-reduce rollback revalidation full sweep failed badly by
  mean timing, while its separate warm 128f confirmation was clean.

This made the previous producer-owner-reduce negative note too strong. The
owner-list ABI is still not promoted, but the available negative speed evidence
was partially confounded by cold setup and MPS timing spikes. Future
owner-reduce work should be a separately named op/mode and should be measured
with the robust gate below before being rejected or promoted.

## Change

Added:

```text
research_experiments/world_foam_lane2/verify_framegroup16_timing_robust.py
research_experiments/world_foam_lane2/test_verify_framegroup16_timing_robust.py
```

The verifier records mean and median first-to-last scales, per-row
mean/median and max/median contamination, storage scale, and optional
max-frame confirmation. A confirmed max-frame artifact can establish
`promoted_path_not_regressed=true`, but the verifier keeps
`clean_speedscale_artifact=false` for the contaminated full sweep.

## Results

Unit gate:

```text
PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust -v
```

Passed: 4 tests.

Accepted loss-reduced artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_framegroup16_timing_robust_lossreduce_accepted.json
```

- `status=ok`
- `clean_speedscale_artifact=true`
- total mean/median scales: `1.464 / 1.532`
- backward mean/median scales: `1.536 / 1.532`
- storage scale: `1.026`

Current revalidated mixed sweep plus warm 128f confirmation:

```text
research_experiments/world_foam_lane2/results/2026-05-16_framegroup16_timing_robust_current_revalidated_confirmed_outlier.json
```

- `status=confirmed_outlier`
- `clean_speedscale_artifact=false`
- `promoted_path_not_regressed=true`
- contaminated 32f row: total max/median `73.85`, backward max/median `85.09`
- contaminated full-sweep 128f median: total/backward `410.28 / 409.83 ms`
- clean 128f confirmation: total/backward median `4.56 / 4.05 ms`
- substituted 16f->128f total/backward median scales: `1.366 / 1.377`

Strict mode on the contaminated mixed sweep exits failed, as intended.

## Takeaway

The promoted framegroup16 loss-reduced path remains the stable baseline. The
later failed full sweep should be classified as outlier-contaminated, not as a
shader regression. Future shader forks need to pass the strict robust verifier
for clean promotion; a confirmed-outlier pass is only enough to avoid reverting
the promoted path.
