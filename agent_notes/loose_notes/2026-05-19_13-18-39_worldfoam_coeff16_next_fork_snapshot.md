# WorldFoam coeff16 next fork snapshot

## Scope

Side-agent pass only. I stayed in the WorldFoam/Gate4 lane and did not touch
STAR UVT files or launch GPU-heavy training.

Read:

- `PROJECT_INDEX.md`
- `TODO/README.md`
- `EXPERIMENTS.md`
- `agent_notes/key_learnings.md`
- latest Gate4/WorldFoam loose notes from 2026-05-19 and 2026-05-18
- `research_experiments/world_foam_lane2/*` scripts and recent result JSONs

## Current read

The latest important correction is that Gate4 coeff16 is a small-MPS keeper
only in the sample-parallel fused-MSE execution path:

- `gate4-affine-candidate-coeff16-fused-mse` keeps selected tape storage nearly
  flat over 2/4/8/16 frames and cuts the previous num32/den16 tape size.
- The track-MSE fork was the wrong execution geometry. It serialized frames in
  one thread per track and was slower even though it reduced atomics.
- Direct-CSR native delta construction is a real CPU/compiler-stage win, but it
  does not change the warm MPS replay kernel.
- Owner-update, ownerkeep, sample-reduce, sortnet, sitecache, framegroup-cache,
  smallrun, packed local-owner, and several native packing/device-residency
  variants are correctness-green or partially useful diagnostics, but not
  promotion keepers.

The best evidence row remains the sample-parallel coeff16 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3.json
```

Its verifier reported `ok`, storage scale about `0.992`, total median scale
about `0.870`, and backward median scale about `0.861`, but the environment was
`contended`. Treat it as directionally useful, not final promotion evidence.

## Best non-GPU next fork

Do not spend the next side-agent turn on another MPS timing ladder while the
machine is contended. The useful non-GPU fork is a compiler/representation
preflight:

1. Keep the sample-parallel coeff16 shader contract as the baseline execution
   shape.
2. Prototype only CPU-side row/tape transforms that could reduce backward
   candidate replay or owner scans without adding a second per-candidate stream.
3. Prove any transform with exact tensor parity against the current direct-CSR
   / cut-array paths before touching Metal.
4. Gate promotion with a clean environment, full train/eval, and reference
   artifact non-regression. Ratio-only or contaminated artifacts are not enough.

Concretely, the next plausible fork is not "more packing" and not "reduce
atomics." It is an owner/boundary/run representation that the sample-parallel
VJP can consume without replaying and re-ownering the full candidate list per
sample. If that representation cannot be built exactly on CPU against the
existing Gate4 affine candidate CSR, it is unlikely to be worth a shader fork.

## Helper added

Added:

```text
research_experiments/world_foam_lane2/summarize_gate4_coeff16_artifacts.py
```

This is intentionally CPU-only. It reads Gate4 coeff16 train/eval JSON
artifacts and emits a compact markdown or JSON table with tape mode,
environment status, scale, 16f timing, storage, and PSNR. The point is to stop
future fork selection from depending on a single loose-note paragraph when the
results directory contains many similar contaminated and paired-control
artifacts.

Validation run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/summarize_gate4_coeff16_artifacts.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/summarize_gate4_coeff16_artifacts.py
```

The summary confirmed the newest coeff16 family is dominated by the
sample-parallel baseline/control rows, with sitecache/framegroup-cache only
diagnostic and track/sortnet/owner variants slower or contaminated.
