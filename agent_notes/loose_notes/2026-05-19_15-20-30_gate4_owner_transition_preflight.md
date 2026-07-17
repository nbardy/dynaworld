# Gate4 owner-transition preflight

## Context

After the densitymask fork rejected another local-cache idea, I stopped adding
Metal variants and tested the one remaining plausible lever on CPU: can the
sample-parallel coeff16 path avoid full owner scans by replaying boundary
transitions exactly?

Added:

```text
research_experiments/world_foam_lane2/analyze_gate4_owner_transition_preflight.py
research_experiments/world_foam_lane2/test_analyze_gate4_owner_transition_preflight.py
```

The analyzer rebuilds the Gate4 affine candidate CSR tape, reconstructs sorted
cut depths per sample, computes authoritative full-scan owners, and compares
three transition policies:

- `ownerkeep`: keep current owner across unrelated boundary ids.
- `ownergroup_keep`: keep current owner across unrelated boundary ids, but keep
  all same-depth boundary ids as a group so duplicate/tie cuts can update or
  force a local fallback.
- `ownerupdate_fallback`: the earlier conservative ownerupdate behavior that
  invalidates the current owner on unrelated cuts and rescans the next segment.

## Validation

Passed:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2 \
  .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/analyze_gate4_owner_transition_preflight.py \
  research_experiments/world_foam_lane2/test_analyze_gate4_owner_transition_preflight.py

rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2 \
  .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_analyze_gate4_owner_transition_preflight -v
```

The unit suite is `4/4`.

Full CPU preflight:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2 \
  .venv/bin/python research_experiments/world_foam_lane2/analyze_gate4_owner_transition_preflight.py \
  --frame-counts 2,4,8,16 \
  --render-size 16 \
  --site-count 24 \
  --sample-validation full \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_owner_transition_preflight_grouped_render16_site24_full_2_4_8_16.json
```

The artifact status is `ok` with zero `missing_sample_events` under authoritative
full validation.

## Results

At render16/site24/full validation:

| frames | baseline owner scans | ownergroup scans | scan reduction | ownergroup exact | plain ownerkeep exact | fallback-update scans |
| ---: | ---: | ---: | ---: | :---: | :---: | ---: |
| 2 | 23,165 | 1,024 | 22.62x | yes | yes | 22,323 |
| 4 | 47,030 | 2,048 | 22.96x | yes | yes | 44,601 |
| 8 | 92,665 | 4,096 | 22.62x | yes | no, 9 mismatches | 87,475 |
| 16 | 186,303 | 8,192 | 22.74x | yes | no, 5 mismatches | 175,634 |

Candidate count stays flat-ish (`84,930 -> 84,225`, scale `0.992x`), but the
per-sample candidate-depth replay iterations still scale with active samples
(`169,860 -> 1,347,600`). Ownergroup scan count also scales exactly with active
samples (`1,024 -> 8,192`), which is expected: one authoritative initial owner
per sample. This does not by itself prove wall-time linearity because the
sample-parallel shader can schedule those samples in parallel, but it does mean
the preflight is a work-reduction result, not a frame-count asymptotic miracle.

The key diagnostic is the huge unrelated-boundary rate:

- ownergroup unrelated boundary fraction is about `94-96%`.
- ownerupdate-fallback is exact, but because it invalidates the current owner on
  those unrelated cuts, it only reduces owner scans by `1.04-1.06x`.
- this explains why the previous ownerupdate/i16 shaders lost despite the
  transition idea being mathematically strong.

Plain ownerkeep in this CPU abstraction almost works, but it is not exact at
8/16f because the collapsed cut array can discard an important same-depth
boundary id. The grouped policy fixes those misses by retaining all boundary ids
at same-depth cuts. In this probe `ambiguous_boundary_groups=0`, so the grouped
path gets exactness without additional fallback scans.

Important nuance: the existing Metal ownerkeep shader replays raw inserted
boundary ids rather than the CPU collapsed cut array, so this preflight should
not be read as "the current ownerkeep shader is wrong." The prior Metal
ownerkeep timing rejection still stands. The useful conclusion is that the
owner-transition math wants keep-unrelated/raw-or-grouped boundary metadata, but
the hot path must make that metadata cheaper than the previous boundary stream.

## Decision

The next Metal fork should not be another field cache, full sort, or conservative
ownerupdate. The preflight points to a cheaper ownerkeep-family shader:

1. Keep the sample-parallel coeff16 launch shape.
2. Keep depth replay, but carry enough raw/grouped boundary metadata to update
   owner exactly across same-depth cuts.
3. Keep owner across unrelated boundary groups instead of invalidating it.
4. Fall back to one scan only for genuinely ambiguous grouped cuts.
5. Prefer an ownerkeep-i16 or otherwise packed side stream first, because the
   ownerupdate-i16 fork reduced dtype width but kept the wrong fallback policy.

Risks:

- This still adds boundary metadata in the hot path, so the earlier ownerupdate
  storage/read penalty remains a real risk.
- It reduces owner scans by `~22.7x`, but not candidate-depth replay.
- Active-sample initial scans scale with frame count; any wall-time sublinearity
  must come from sample-parallel MPS scheduling, as with the current keeper.
