# WorldFoam G4-v2 spatial heldout and paper-asset completion

## User correction and evidence boundary

The user correctly rejected describing source/unit checks as the goal.  The
paper deliverables are measured ablations:

- G4 public quality remains `0/36` measured rows.
- G6 native memory/work remains `0/21` measured rows, plus zero of the three
  auxiliary restart processes.
- `native_memory_fit=false` currently means not measured, not measured failure.

No native rebuild, MPS dispatch, dataset-cache construction, or publication
row ran in this work chunk.  The host remained unsuitable after the prior
memory incident.  The 8-GiB available-RAM launch guard is incident headroom,
not a claim that WorldFoam needs 8 or 32 GiB.  The analytic `S=1024` state is
still `114,688 B` live, `81,920 B` checkpoint, and `278,528 B` for the
conservative live-plus-checkpoint-payload-clone peak.  Allocator, native
scratch, compiler, mapped-file, and process-group memory still require G6.

## G4-v2 implementation truth

The old all-pixel G4-v1 schedule remains an exact reference and remains
computationally infeasible: roughly 113--115 million cold `(view,pixel)`
compiles per seed.  It was not weakened or silently mutated.

The separately versioned G4-v2 selected-ray protocol is now source-wired:

- three scenes x seeds `17/29/43` x four routes = 36 rows;
- 300 optimizer steps x four spacetime samples x 1024 shared selected pixels;
- `1,228,800` target pixels and `3,686,400` RGB-MSE scalars per row;
- one route-independent RGB-MSE contract;
- exact selected target-read ownership receipts;
- WorldFoam rasterized training work = `1,228,800` pixels;
- Gaussian control rasterized training work = `235,929,600` pixels;
- one child-inclusive 4-GiB process-group watchdog/receipt per row; and
- an independent pure collected-artifact verifier for the exact 36-row grid.

Target pixel budget is matched; rasterized work is deliberately not claimed
equal and must be a paper column.

## Spatial-major full-temporal heldout evaluation

The old frame-major WorldFoam heldout loop would compile one track per time and
materialize unnecessary ray traffic.  The new evaluator compiles a spatial
track across all 300 frames once:

- `196,608` cold track compiles per one-camera `384x512` heldout sequence;
- `58,982,400` complete camera-record validations;
- `1,536` host calls of 128 tracks;
- `15,360` internal native bundles bounded to 4096 observations;
- no mmap, no heldout ray construction, and no dense device video;
- bounded 1024-track CPU write superblocks;
- one frame-major prediction-f32 spool (`675 MiB`);
- one frame-major target-RGB8 spool (`168.75 MiB`);
- `843.75 MiB` total temporary disk, explicitly not representation RAM;
- sequential completed-file hashes plus per-frame source-to-spool target hashes;
- exact PSNR, SSIM, LPIPS, L1, media, and coverage order;
- explicit accounting of 2T public/cache target reads and 3T literal target
  traversals; and
- deletion proof on success and failure.

Full-coverage and bounded-pilot receipts are distinct.  Focused CPU validation
reported `5 passed`; this is preflight, not an ablation row.

## G6 dry-plan truth

The allocation-free clean-host command was rerun:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  research_experiments/world_foam_lane2/run_worldfoam_g6_clean_host_bundle.py
```

It reported:

- `blocking_reasons=[]`;
- status `static_prebuild_ready_host_unchecked`;
- exact `12 + 9 = 21` measured-row plan plus three restart processes;
- hard 2-GiB MPS and 4-GiB process-group RSS gates;
- no Torch import, native import, build, subprocess, artifact, MPS allocation,
  or evidence row; and
- one expected pre-build condition: the installed native binary is older than
  the bound 133-schema native sources.  The real `--execute` route force-builds
  and attests it before measuring.

## Paper tables and figures

The presentation path no longer stops at permanent G4/G6 placeholders:

- `generate_worldfoam_public_quality_assets.py` consumes verifier-accepted
  G4-v2 evidence and emits CSV, TeX, JSON, and a scene/route PSNR SVG.  It
  reports rasterized work and preserves compatibility with historical G4-v1
  asset fixtures without allowing v1 into the v2 evidence gate.
- `generate_worldfoam_training_memory_assets.py` consumes only an accepted
  21-row G6 artifact and emits CSV, TeX, JSON, and an SVG showing sampled MPS
  high water and normalized ordered-word work across `F=8/64/300`.
- `generate_worldfoam_paper_b_artifacts.py` now includes G4/G6 as independent
  evidence specs.  Missing/rejected gates retain `NOT MEASURED`; accepted exact
  matrices replace the placeholders and promote only their supported claims.
- `verify_worldfoam_iclr_package.py` now uses the G4-v2 verifier.

The current incomplete foundation bundle was regenerated and remains honest:
G4/G6 are missing, the two placeholders remain, and the promoted claims remain
false.  Safe CPU presentation gates reported `35 passed` after regeneration.

## Next execution order

1. Finish source integration of the spatial evaluator wrapper into G4-v2 raw
   rows, verifier, and capability receipt.
2. Produce the bounded Coffee Martini seed-17 WorldFoam pilot: one real 4096-
   target optimizer step, a bounded 300-time heldout block, bitwise old/new
   parity, measured compile/work/memory/timing, and explicit
   `public_quality_evidence=false`.
3. On a clean capable host, run all 36 G4 rows.
4. On a quiet Apple-silicon Mac, run the G6 clean-host `--execute` bundle.
5. Regenerate the Paper-B bundle; only accepted artifacts may replace the G4
   and G6 placeholders.

